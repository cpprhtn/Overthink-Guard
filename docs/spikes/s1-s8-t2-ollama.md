# S1 · S8 · Tier 2: Ollama 실측

- 날짜: 2026-09-29
- 환경: macOS (Apple Silicon), Ollama 0.34.4 (Homebrew), `qwen3:1.7b`
- 스크립트: `bench/spikes/` (Ollama가 `localhost:11434`에서 떠 있어야 함)

## S1. thinking 스트리밍 형식과 종료 주입

### 스트리밍 형식

Ollama는 `<think>` 태그를 제거하고 thinking을 **별도 필드**로 보낸다. 태그 파서(`ThinkStreamParser`)는 llama.cpp 등 태그가 남는 백엔드용이고, Ollama에는 필드 기반 경로가 필요하다.

| 엔드포인트 | thinking | answer |
| --- | --- | --- |
| `/api/chat` (네이티브) | `message.thinking` | `message.content` |
| `/v1/chat/completions` | `choices[].delta.reasoning` (`reasoning_content` 아님) | `choices[].delta.content` |

네이티브 응답의 마지막 줄에는 `prompt_eval_cached_count`가 있어 prefix 캐시 적중을 잴 수 있다.

### 종료 주입: raw가 아니라 assistant prefill

Ollama의 qwen3 템플릿은 **마지막 메시지가 assistant이면 턴을 닫지 않고**, `thinking` 필드를 `<think>{{ .Thinking }}</think>`로 렌더링한다. 따라서 아래 요청을 보내면 Ollama가 템플릿을 직접 적용한 상태로 모델이 곧바로 답변을 이어 쓴다.

```json
{"model": "qwen3:1.7b", "think": true, "messages": [
  {"role": "user", "content": "<질문>"},
  {"role": "assistant", "thinking": "\n<받은 thinking>\n\nI have enough to answer now.", "content": ""}
]}
```

- 새 thinking 0자, 곧바로 답변. 40% 지점에서 주입한 두 번의 시도 모두 정답(45).
- **chat template을 재구현할 필요가 없다** (검토 2번 해소, 단 이 템플릿 구조를 가진 모델 한정).
- 손으로 만든 raw 프롬프트(`/api/generate`, `raw: true`)도 동작은 한다. 하지만 `think: true`일 때 템플릿이 사용자 메시지 끝에 ` /think`를 붙이는 것까지 재현해야 캐시가 맞는다. 손으로 재현하는 방식은 깨지기 쉽다.

### 중단과 캐시 재사용

- 클라이언트가 연결을 끊으면 서버가 생성을 멈춘다 (로그: `stop: cancel task`).
- 끊은 직후 prefill하면 thinking을 **받은 그대로** 넣었을 때 캐시가 거의 재사용되지 않는다. Ollama가 `<think>` 뒤의 줄바꿈을 떼어 내서 토큰이 어긋나기 때문이다. **앞에 `"\n"`을 붙이면** 원래 생성 시퀀스와 정렬된다.

| prefill thinking | prompt 토큰 | 캐시 적중 | prompt 처리 |
| --- | --- | --- | --- |
| 받은 그대로 | 777 | 24 | 763ms |
| `"\n"` + 받은 것 | 800 | 790 | 48ms |
| raw `<think>\n` + 받은 것 (` /think` 누락) | 770 | 16 | 764ms |

이 `"\n"` 보정은 모델·템플릿마다 다를 수 있으므로 템플릿 YAML의 `stop_injection.thinking_prefix` 필드로 두었다.

**프록시가 쓰는 조합 (A')**: 원래 요청은 `think`를 지정하지 않고 클라이언트 요청 그대로 보내고, 끊은 뒤의 prefill에만 `think: true`를 붙인다. `think`를 지정하면 템플릿이 사용자 메시지에 ` /think`를 붙이지만, 이 조합에서도 캐시는 826개 중 816개가 적중했다. 대안인 "`think` 없이 assistant `content`에 `<think>\n…</think>\n\n`을 직접 넣기"도 동작했다(814개 중 804개 적중, 출력은 `content`로 분류됨). 하지만 태그 문자열에 의존하므로 채택하지 않았다.

### 토큰 수

Ollama 네이티브 스트림은 **청크 하나에 토큰 하나**다(S8에서 청크마다 logprob 항목이 1개). 프록시는 청크 수를 실제 토큰 수로 쓴다. 실측에서 UI 표시 964, `eval_count` − 답변 청크 = 970이었다. 반면 텍스트 추정(ASCII 4자당 1토큰)은 Qwen의 수학 thinking(약 2.3자당 1토큰)을 **약 1.7배 적게 센다**. 오프라인 `otg analyze`와 `bench/r1_replay.py`의 절대 토큰 수(`min_thinking_tokens` 같은 임계값 비교 포함)는 이만큼 부정확할 수 있다. 비율(절감률)은 영향이 작다.

### 프록시 end-to-end (`otg start`, OpenAI Python 클라이언트, base_url만 교체)

| 시나리오 | 결과 |
| --- | --- |
| 개입 없음 | reasoning·content 정상 수신, 정답, usage는 Ollama 값 그대로 |
| thinking 300청크 뒤 "Answer now" | 누른 뒤 **0.09초** 만에 첫 답변 토큰, 캐시 337개 중 327개 적중, 정답. 생성 토큰 809개 (개입 없을 때 3,193개) |
| `/v1/models`, `/api/tags` 통과 | 정상 |
| `Host: evil.example`, `Origin: https://evil.example` | 403 |

끊긴 요청은 Ollama가 `prompt_eval_count`를 주지 않는다. 그래서 개입한 응답의 `usage.prompt_tokens`는 "답변 요청의 prompt − 끊기 전 생성 토큰"으로 추정한다(실측 35, 실제 23). 차이는 주입 문구 토큰 때문이다.

## S8. 스트리밍 logprob과 종료 토큰 마진 (Tier 1)

- `logprobs: true, top_logprobs: N`이면 `/api/chat`과 `/v1` 모두 **스트리밍 중 토큰마다** logprob과 top-k를 준다. top-k에 `</think>`가 토큰으로 나타난다.
- 하지만 **실제로 생성된 `</think>` 위치의 logprob 항목은 스트림에서 빠진다** (Ollama가 태그 파싱 과정에서 먹는다).
- 신호 자체도 약했다. LCM 문제의 thinking 1,374토큰 중 `</think>`가 top-20에 든 위치는 첫 토큰 1곳뿐이었다. 정답을 아는 8문제에서 문장 경계 마진 임계값 τ ∈ {1, 2, 3, 5}로 **한 번도 발동하지 않았다.**
- 결론: **qwen3:1.7b에서는 종료 토큰 마진이 조기 신호를 주지 않는다.** 모델은 끝내기로 결정하는 순간까지 `</think>`에 거의 확률을 주지 않는다. 더 큰 모델이나 다른 계열에서는 다를 수 있다.

## Tier 2. 능동 탐침

thinking의 10%, 20%, …, 90% 지점에서 탐침했다. 방법은 prefill(`thinking` + `content: "The final answer is $\boxed{"`, `num_predict: 12`)로 답만 짧게 뽑는 것이다. 이 탐침은 사후 시뮬레이션이다(전체 생성 후 잘라서 탐침).

| 문제 | 정답 | 탐침 답 (10%→90%) | 처음 정답 |
| --- | --- | --- | --- |
| 3600의 약수 개수 | 45 | 전부 45 | 10% |
| 처음 50개 홀수의 합 | 2500 | 전부 2500 | 10% |
| 180km / 2.5h | 72 | 전부 72 | 10% |
| 5명 일렬 배치 | 120 | 전부 120 | 10% |
| 2^10 mod 7 | 2 | 1,1,1,1,2,2,2,2,2 | 50% |
| 3x+7=25 | 6 | 4,6,6,… | 20% |
| 12와 18의 LCM | 36 | 전부 36 | 10% |
| 30 미만 소수 개수 | 10 | 10,10,15,10,… | 10% |

- 연속 2회 일치하면 종료: **8/8 발동, 7/8 정답, 생성 토큰 80% 절감**(탐침 비용 제외).
- 실패 1건(2^10 mod 7)은 탐침이 오답에 먼저 수렴한 경우다. 연속 3회로 바꿔도 30%에서 오답으로 끊는다. R3(성급한 종료)의 실례다.
- 탐침 비용: 캐시 정렬 덕분에 문제당 9회 탐침 합계 2~4초. 예외적으로 thinking이 아주 길었던 1건은 22.8초가 걸렸다.

## 종합과 설계 영향 (오너 결정 필요)

| 신호 | 결과 | 판정 |
| --- | --- | --- |
| Tier 0 텍스트 (R1 986 trace, `tier0-r1-replay.md`) | 안전한 설정은 절감 ≤2%, 25%를 넘기면 위험 약 24% | 로컬 Auto의 주 신호로는 no-go |
| Tier 1 종료 토큰 마진 (qwen3:1.7b) | 8문제 모두 미발동 | 이 모델에서는 no-go |
| Tier 2 능동 탐침 (qwen3:1.7b) | 7/8 정답, 80% 절감 | 유망. 표본이 작고 쉬운 문제뿐 |

- **D12는 로컬 Auto에 대해 뒤집어야 할 가능성이 높다.** 문서가 기각한 "능동 탐침 기본값"이 실측으로는 유일하게 동작한 신호다. D12를 버린 이유였던 R4(탐침 오버헤드)는 prefill 캐시 정렬로 이 환경에서는 대부분 해소되었다. CPU 전용 노트북에서는 아직 확인하지 않았다.
- Tier 0은 API·구독형 관측, UI 국면 표시, 그리고 **탐침을 언제 쏠지 고르는 트리거**(예: 잠정 결론 표현이 나온 문장 경계)로 역할을 바꾸는 것을 제안한다.

## 실시간 탐침 (스크립트: `bench/spikes/rt_probe.py`)

3600의 약수 문제, 400토큰마다 탐침. 기준 생성 속도는 77 tok/s였다.

| 방식 | 결과 |
| --- | --- |
| P: 본 생성과 병렬로 탐침 | **사용 불가.** 기본 설정의 Ollama는 요청을 하나씩 처리한다. 탐침은 본 생성이 끝날 때까지 대기열에 머물렀다(대기 34초 → 7초로 감소). 사용자에게 `OLLAMA_NUM_PARALLEL` 변경을 요구할 수는 없다. |
| S: 멈춤 → 탐침 → 재개 | **동작.** 탐침 1회 0.2초, 캐시 거의 전부 적중. 8회 탐침 포함 총 시간 +3% (C3 ≤10% 충족). 탐침 답은 800토큰부터 정답으로 수렴. |

재개 방식: chat의 `thinking` 필드는 블록을 항상 `</think>`로 닫기 때문에 재개에 쓸 수 없다. 대신 `think` 없이 assistant `content`에 `"<think>\n" + 받은 thinking`을 넣는다.
- 첫 청크까지 0.04초, 캐시 251개 중 250개 적중.
- 이어지는 출력은 **`content`로 분류**된다. `</think>`는 청크 하나로, 그 뒤 `"\n\n"`과 답변이 온다.
- 그래서 재개 후에는 태그 파서(`ThinkStreamParser`, `starts_in_thinking=True`)로 thinking과 답변을 나누고, 답변 앞 공백을 제거한다.

클라이언트가 `seed`를 지정하면 재개 시 샘플링이 달라져 재현성이 깨진다. 그런 요청은 탐침하지 않는다.

## CPU 전용 탐침 비용 (`bench/spikes/cpu_overhead.py`)

Ollama 옵션 `num_gpu: 0`으로 GPU 없이 생성했다. 문제는 3600의 약수, thinking 2,000토큰, 400토큰마다 탐침.

| 항목 | 결과 |
| --- | --- |
| 기준 생성 속도 | 33 tok/s (Apple Silicon CPU) |
| 탐침 1회 | 0.5~1.0초, 캐시 적중은 GPU와 같음 (예: 1225/1242) |
| 탐침 시간 합계 / 기준 실행 시간 | 3.1초 / 60.5초 = **약 5%** |
| 전체 시간 차이 | +1.6%. 실행마다 생성 길이가 흔들리는 잡음 범위라 위 5%를 오버헤드로 본다 |

탐침 비용은 짧은 디코딩(최대 16토큰)과 캐시 밖의 주입 문구(약 17토큰)가 대부분이다. 그래서 생성 속도가 느린 기기에서도 비율이 크게 변하지 않을 것으로 본다. 다만 x86 노트북 CPU는 측정하지 않았다.

## 다른 모델: deepseek-r1:1.5b에서는 prefill 방식이 동작하지 않는다

Ollama 0.34.4의 `deepseek-r1:1.5b` 템플릿에는 세 가지 특징이 있다.
- assistant 메시지에서 `</think>` 앞부분을 잘라 버린다.
- `thinking` 필드를 렌더링하지 않는다.
- 마지막 assistant 턴도 항상 문장 끝 토큰으로 닫는다.

실측 결과:

| 시도 | 결과 |
| --- | --- |
| 지금 답해 (`thinking` 필드 prefill, `think: true`) | prompt 17토큰. thinking이 통째로 버려졌다. 출력은 "Provide your answer as a number… Wait, I just thought of something…" 같은 엉뚱한 텍스트 |
| 재개 (`content`에 `"<think>\n"`+thinking) | 이어 쓰지 않고 `<think>`부터 새로 생각을 시작했다 |
| raw 모드 `/api/generate` (템플릿을 손으로 재구성) | 답변 경로는 정상적인 답을 시작했다. 재개도 이어서 생각했고 캐시 146/147 적중. 단, 모델별 템플릿 문자열을 직접 관리해야 한다 |

**조치**: 템플릿에 `prefill_supported`를 추가했다. 실측으로 확인된 qwen3만 `true`이고, deepseek_r1과 알 수 없는 모델(generic)은 `false`다. `false`인 모델에서는 다음과 같이 동작한다.
- "지금 답해"는 거절된다(UI 버튼 비활성).
- 탐침은 하지 않는다(`probe_skipped: template`).
- Shadow Tier 0 관찰만 한다.

엉뚱한 답을 내는 것보다 개입하지 않는 쪽이 안전하다(C1, C7).

### 후속: raw 모드로 deepseek-r1 지원 (2026-09-30)

스크립트: `bench/spikes/r1_raw_spike.py`, `bench/spikes/r1_live_check.py`

**raw 프롬프트 검증**
- `<｜User｜>{질문}<｜Assistant｜>` 형식을 BOS 없이 보내면, Ollama chat 렌더링과 prompt 토큰 수가 같다(18 = 18). 시스템 메시지가 있을 때도 같다(18 = 18).
  - Ollama가 BOS를 스스로 붙이므로, BOS 문자열을 넣으면 오히려 1토큰 많아진다.
- `/api/generate`에 `raw: true`로 보내도 Ollama는 thinking을 `thinking` 필드로 분리한다.
- 주입과 재개에는 `"<think>\n"`을 다시 붙여야 한다. 그러면 원래 생성 시퀀스와 정렬되어 캐시가 맞는다(418/428). 빼면 캐시가 깨진다(18/426). qwen3의 `"\n"` 정렬과 같은 원리다.
- 지금 답해의 답변은 `response` 필드로 온다. 재개의 이어 쓰기도 `response`에 태그와 함께 온다(qwen3 재개와 같은 모양).

**구현**: 템플릿에 `raw_prompt`(system/user/assistant 형식)를 두면 `RawPrompt` 요청 방식을 쓴다. 이때 원래 요청부터 raw로 보내 캐시를 공유한다. `prefill_supported`나 `raw_prompt`가 있으면 개입할 수 있다(`Template.can_intervene`).

**프록시 실측** (deepseek-r1:1.5b)

| 시나리오 | 결과 |
| --- | --- |
| 탐침 켜고 끝까지 | 탐침 0.17초, 답 `36`. 재개 후 이어서 생각하고 정답(36) |
| 150청크 뒤 지금 답해 | 0.13초 만에 첫 답변, 캐시 169/179, 정답(45) |

**주의**: raw 형식은 모델 템플릿을 손으로 옮긴 것이다. Ollama가 모델의 템플릿을 바꾸면 어긋날 수 있다. 시스템 메시지가 여러 개면 마지막 것만 쓴다(원래 템플릿의 동작과 같다). 멀티턴은 원래 개입하지 않는다(D9).

## 아직 확인하지 않은 것

1. ~~더 크고 어려운 문제 세트~~ → `shadow-live-probe.md`(37문제)에서 측정했다. 다른 모델: deepseek-r1은 prefill 불가로 확인. gpt-oss와 더 큰 qwen3는 미확인.
2. ~~CPU 전용 탐침 비용~~ → Apple Silicon CPU에서 약 5%. x86 노트북은 미확인.
3. S2(llama.cpp), S3(LM Studio).
4. ~~deepseek-r1 계열을 위한 raw 모드 백엔드~~ → 구현하고 deepseek-r1:1.5b에서 확인했다(위 "후속" 절). 더 큰 R1 distill(7b, 8b 등)은 미확인이지만 템플릿은 같다.
