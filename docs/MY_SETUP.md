# 내 StackChan + OpenRouter 세팅 가이드

이 문서는 Ernest 개인용 세팅 기록입니다. 원본 프로젝트(rudyll/stackchan-ai-server)를 fork한
이 저장소에 두고, 실제 진행 상황과 값을 채워 나가는 용도로 씁니다.

주의: 이 파일에는 실제 API 키를 절대 적지 마세요. 키는 서버를 실행하는 컴퓨터의 .env 파일에만
넣습니다. .env는 .gitignore에 포함되어 있어 GitHub에는 절대 올라가지 않습니다.

## 0. 전체 구조 한 줄 요약

[StackChan 로봇] --WiFi--> [중계 서버: stackchan-ai-server] --API--> [OpenRouter(LLM) + STT/TTS]

- StackChan은 마이크/스피커 역할만 합니다.
- 중계 서버가 음성을 텍스트로 바꾸고(STT), OpenRouter에 보내 답을 받고(LLM), 다시 음성으로
  바꿔(TTS) StackChan에 돌려줍니다.
- 서버는 웹소켓으로 StackChan과 항상 연결되어 있어야 하므로 "상시 켜져 있는 컴퓨터"가 필요합니다.

## 1. 지금 단계 체크리스트

- [x] 하드웨어 확인: M5StackChan (ESP32-S3, XiaoZhi 엔진 기반)
- [x] rudyll/stackchan-ai-server fork 완료 -> ERNESTLOOK/stackchan-ai-server
- [ ] OpenRouter 계정 생성 + API 키 발급
- [ ] STT(음성인식)/TTS(음성합성) API 키 준비 (OpenAI 권장)
- [ ] 서버를 돌릴 컴퓨터/서버 결정 (지금은 내 컴 로컬 -> 나중에 맥미니로 이전)
- [ ] Docker로 서버 실행 + 웹 설정 화면에서 provider=openrouter 지정
- [ ] StackChan에 서버 주소(NVS) 기록
- [ ] 대화 테스트

## 2. API 키 발급

### OpenRouter (LLM 두뇌)
1. https://openrouter.ai 접속 -> 회원가입
2. 우측 상단 계정 메뉴 -> Keys -> Create Key
3. 발급된 키를 안전한 곳(비밀번호 관리자 등)에 저장 -- 여기 문서에는 적지 않기
4. Credits 메뉴에서 소액 충전 (모델마다 과금 방식 다름, $5~10 정도로 시작 추천)
5. 쓰고 싶은 모델명 후보 기록:
   - anthropic/claude-3.5-sonnet
   - openai/gpt-4o
   - (OpenRouter 모델 목록: https://openrouter.ai/models)

### STT/TTS (음성 인식/합성)
OpenRouter는 텍스트만 처리하므로 별도로 필요합니다. 가장 간단한 조합은 OpenAI:
1. https://platform.openai.com 에서 API 키 발급
2. STT 모델: whisper-1
3. TTS 모델: tts-1, 음성: alloy 등

## 3. 서버 실행 (내 컴퓨터, 로컬)

전제: Docker Desktop 설치됨.

    git clone https://github.com/ERNESTLOOK/stackchan-ai-server.git
    cd stackchan-ai-server/stackchan-server
    cp .env.standalone.example .env

.env 파일을 열어 아래 값을 채웁니다 (실제 키/IP는 이 저장소가 아니라 내 컴퓨터의 이 파일에만):

    STACKCHAN_LOCAL_HOST=<내 컴퓨터의 로컬 IP, 예: 192.168.0.15>
    STACKCHAN_AI_PROVIDER=openrouter
    STACKCHAN_OPENROUTER_API_KEY=<OpenRouter 키>
    STACKCHAN_LLM_MODEL=anthropic/claude-3.5-sonnet
    STACKCHAN_STT_BASE_URL=https://api.openai.com/v1
    STACKCHAN_STT_API_KEY=<OpenAI 키>
    STACKCHAN_STT_MODEL=whisper-1
    STACKCHAN_TTS_BASE_URL=https://api.openai.com/v1
    STACKCHAN_TTS_API_KEY=<OpenAI 키>
    STACKCHAN_TTS_MODEL=tts-1
    STACKCHAN_TTS_VOICE=alloy

실행:

    docker compose -f docker-compose.standalone.yml up --build -d

웹 설정 화면 접속: http://localhost:8099 (또는 .env에서 지정한 포트)

## 4. StackChan에 서버 주소 알려주기 (NVS 주입)

1. StackChan을 USB-C로 컴퓨터에 연결
2. 저장소 루트의 flash_nvs.py 실행:

       python3 flash_nvs.py

3. 프롬프트에서 서버 IP(위 STACKCHAN_LOCAL_HOST 값)와 포트(기본 12800) 입력
4. StackChan 재부팅 -> Wi-Fi로 서버에 자동 연결

## 5. 나중에: 맥미니로 이전할 때

- 맥미니에 Docker 설치 -> 이 저장소를 clone -> 같은 .env 값으로 재실행
- StackChan의 서버 IP도 맥미니의 IP로 다시 NVS 주입 (flash_nvs.py 재실행)
- 맥미니를 상시 켜두면(취침모드 방지 설정) 내 컴을 껐다 켜도 StackChan은 항상 응답 가능

## 6. 진행 메모 (여기에 계속 업데이트)

- 2026-09-06: 기기 확인(M5StackChan ESP32-S3), 저장소 fork 완료.
- 2026-09-07: 이브(Eve) 운영 기준을 "기본 활기 + 더 똑똑한 AI"로 고정.
  - 몸동작: 기본 펌웨어의 idle/blink/breath/touch/speaking 루프를 우선 사용.
  - 서버 AI: 한국어 대화, 기억, 명시 요청 기반 카메라/동작 도구만 담당.
  - MCP 도구: LLM에는 저수준 서보/LED 대신 `stackchan_see`, `stackchan_face`, `stackchan_move`, `stackchan_nod`, `stackchan_shake`, `stackchan_status`, `stackchan_health` 같은 고수준 행동만 노출.
  - 카메라: "카메라", "사진", "촬영", "찍어", "봐줘"처럼 명시적인 시각 요청이 있을 때만 사용.
  - 말투: 교수님 호칭, 자연스러운 한국어, 보통 1-2문장, 확인하지 못한 사실은 추측하지 않기.
  - 커스터마이징: `stackchan_motion_speed`, `stackchan_motion_step_delay_ms`, `stackchan_expression_colors`로 동작 속도와 표정별 LED 색을 조정.
