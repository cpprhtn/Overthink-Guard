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

이 `"\n"` 보정은 모델·템플릿마다 다를 수 있으므로 템플릿 YAML의 필드로 두어야 한다 (컨트롤러 구현 시 추가).

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

## 아직 확인하지 않은 것

1. **실시간 탐침**: 위 탐침은 전체 생성 후 사후에 잘라서 한 것이다. 실시간으로 하려면 둘 중 하나가 필요하다. ① 본 생성과 탐침을 병렬로 돌린다(`OLLAMA_NUM_PARALLEL`, 슬롯 간 캐시 공유 여부 미확인). ② 멈춤 → 탐침 → 재개. chat prefill은 thinking을 `</think>`로 닫아 버리므로 재개에는 쓸 수 없고, raw 모드로 템플릿을 정확히 재현해야 한다.
2. 더 크고 어려운 문제 세트(탐침 7/8은 쉬운 문제 8개 기준), 다른 모델(deepseek-r1 distill, gpt-oss).
3. CPU 전용 노트북에서의 탐침·재처리 비용(C3).
4. S2(llama.cpp), S3(LM Studio).
