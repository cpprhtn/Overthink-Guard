# S4: Claude Code(구독)에서 무엇을 실시간으로 볼 수 있나

- 날짜: 2026-09-30
- 환경: macOS, Claude Code 2.1.233 (VS Code 확장과 CLI), 구독 로그인
- 원칙(D5): 인증 정보는 읽지 않고, 요청을 대신 보내거나 프록시하지 않는다. 사용자에게 이미 보이는 것과 로컬 기록만 읽는다.

CLI 2.1.233은 기본 모델(claude-fable-5-1)을 지원하지 않아(2.1.251 이상 필요) 스파이크는 `--model sonnet`으로 했다. 사용자의 CLI는 업데이트하지 않았다.

## 1. 헤드리스: `claude -p --output-format stream-json --include-partial-messages`

스크립트: `bench/spikes/s4_headless.py`

| 질문 | thinking 토큰 | thinking 이벤트 | 내용 |
| --- | --- | --- | --- |
| 3600의 약수 개수 | 66 | 2회, 답변 시작과 같은 시각 | 비어 있음 |
| 5가 2018개인 수를 13으로 나눈 나머지 | 961 | 8회, 4.1~12.4초(답변은 12.4초부터) | 비어 있음 |
| x² − 7y² = 1의 해 | 1,443 | 11회, 8.7~20.8초 | 비어 있음 |

- `thinking_delta` 이벤트는 **생각하는 도중에 실시간으로** 온다. 하지만 `thinking` 텍스트는 항상 비어 있고 `signature_delta`만 데이터를 가진다.
- thinking 토큰 수는 끝난 뒤 `result.usage.output_tokens_details.thinking_tokens`로 나온다.
- 공식 문서 조사: main 에이전트의 thinking 텍스트를 stream-json에 넣는 옵션은 없다. `showThinkingSummaries`는 대화형 화면용이고, `--forward-subagent-text`는 서브에이전트 전용이다.

## 2. 대화형: 세션 기록 `~/.claude/projects/<project>/<session>.jsonl`

- 응답은 **콘텐츠 블록마다 별도 레코드**로 기록된다. thinking 레코드는 그 블록이 끝난 뒤 쓰인다(다음 블록까지 중앙값 0.64초). 그래서 기록 파일로 thinking을 읽는 시점에는 이미 생각이 끝났다.
- thinking 블록 중 텍스트(요약)가 있는 것은 일부다(이 세션 447개 중 152개). 나머지는 서명만 있다.
- 레코드마다 `usage.output_tokens_details.thinking_tokens`, `effort`, `timestamp`가 있다. 한 응답이 여러 레코드에 같은 usage로 반복되므로 메시지 ID로 중복을 제거해야 한다.
- 공식 문서는 이 파일 형식이 내부용이며 버전마다 바뀔 수 있다고 경고한다.

## 3. 실시간으로 쓸 수 있는 신호: 침묵 시간

프롬프트나 도구 결과(user 레코드) 뒤에 assistant 레코드가 없는 동안에는 모델이 생각하거나 첫 블록을 생성하는 중이다. 이 PC의 전체 세션 9,741턴 기준으로 다음과 같다.

| 중앙값 | p90 | p99 | 60초 초과 | 90초 초과 |
| --- | --- | --- | --- | --- |
| 6.7초 | 21.9초 | 53.2초 | 0.8% | 0.3% |

- 60초 넘는 침묵 77건 중 직전 레코드는 프롬프트 29건, 도구 결과 47건이었다. 모델 호출이 없는 로컬 명령(`/model`, `!` 셸, 중단 표시, isMeta)은 기다림으로 치지 않는다.
- 생각 도중에 발생하는 공식 훅은 없다(`UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop` 등 경계 시점뿐).

## 4. 세션 상태 파일 `~/.claude/sessions/<pid>.json`

- 대화형 세션만 있다(헤드리스 `claude -p` 실행 중에는 생기지 않았다).
- 필드는 `pid`, `sessionId`, `kind`(`interactive`), `entrypoint`, `status`(`busy`/`idle`, 턴 단위) 등이다. 같은 폴더의 `*.key` 파일은 열지 않는다.
- 프로세스 생존 여부와 `status`로 "끝난 세션"이나 "idle 세션"의 오탐을 막고, 훅 이벤트의 `run_mode`를 채운다.

## 5. 결론과 구현 (R14 판정)

- **생각 도중의 thinking 내용**: 공식 경로도 기록 파일도 주지 않는다. 그래서 Tier 0 판정을 실시간으로 할 수 없다.
- **생각이 길어지고 있다는 사실**: 침묵 시간과 세션 상태로 실시간으로 알 수 있다.
- 그래서 구독형은 다음 두 가지로 구현했다(`otg claude-code watch`, `otg claude-code report`).
  1. **알림**: 기준(기본 60초)을 넘는 침묵이면 데스크톱 알림과 훅 이벤트(6.2 스키마)를 보낸다. 중단은 사용자가 Esc로 한다(D7).
  2. **사후 리포트**: effort별 thinking 토큰과 전체 출력 중 thinking 비중. 최근 7일 기준 응답 859개, thinking 비중 26%였다.

### 실측 확인

- `claude -p`로 thinking이 약 16초 걸린 질문: 관찰기(5초 기준)는 시작 약 7초 뒤, **답변이 시작되기 약 9초 전**에 알렸다.
- thinking이 5초 안에 끝난 질문: 알림 없음.
- 이 VS Code 세션: 5초 넘게 생각한 턴에서 `run_mode: interactive`로 알림.
- 훅이 이벤트 JSON을 받았고, 세션 텍스트는 들어 있지 않았다. macOS 데스크톱 알림도 떴다.

### 알려진 한계

- **헤드리스 실행이 턴 중간에 비정상 종료되면**(예: `--max-turns`에 걸려 도구 결과만 남고 종료) 상태 파일이 없어서 끝난 것을 알 수 없다. 그래서 잘못된 알림이 한 번 갈 수 있다(실측에서 1회 확인).
- 기록 파일과 상태 파일은 Claude Code의 내부 형식이라, 업데이트로 바뀌면 관찰기가 조용히 동작을 멈출 수 있다. `type`, `timestamp`, `usage`, `effort`, `isMeta`/`isSidechain`만 쓰도록 최소화했다.
- 침묵 시간에는 첫 블록 생성 시간도 섞인다. 매우 긴 도구 호출 인자를 쓰는 경우 등이다.
