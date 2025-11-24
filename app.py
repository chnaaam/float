"""
FastAPI Application for FLOAT Inference
"""

import os
import tempfile
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from generate import InferenceAgent, InferenceOptions


# FastAPI 앱 초기화
app = FastAPI(
    title="FLOAT Inference API",
    description="Audio-driven facial animation using FLOAT model",
    version="1.0.0"
)

# 전역 변수로 InferenceAgent 저장
agent = None


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 모델 로드"""
    global agent
    
    # 옵션 설정
    opt = InferenceOptions().parse()
    opt.rank = 0
    opt.ngpus = 1
    
    # InferenceAgent 초기화
    agent = InferenceAgent(opt)
    
    # 결과 디렉토리 생성
    os.makedirs(opt.res_dir, exist_ok=True)
    
    print("✓ FLOAT 모델이 성공적으로 로드되었습니다.")


@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {
        "status": "running",
        "message": "FLOAT Inference API is running"
    }


@app.post("/inference")
async def inference(
    ref_image: UploadFile = File(..., description="참조 이미지 파일"),
    audio: UploadFile = File(..., description="오디오 파일"),
    a_cfg_scale: float = Form(2.0, description="오디오 CFG 스케일"),
    r_cfg_scale: float = Form(1.0, description="참조 CFG 스케일"),
    e_cfg_scale: float = Form(1.0, description="감정 CFG 스케일"),
    emo: Optional[str] = Form(None, description="감정 (angry, disgust, fear, happy, neutral, sad, surprise)"),
    nfe: int = Form(10, description="Number of function evaluations"),
    no_crop: bool = Form(False, description="얼굴 크롭을 하지 않음"),
    seed: int = Form(25, description="랜덤 시드"),
):
    """
    오디오 기반 얼굴 애니메이션 생성 엔드포인트
    
    Args:
        ref_image: 참조 이미지 (jpg, png 등)
        audio: 오디오 파일 (wav, mp3 등)
        a_cfg_scale: 오디오 조건 스케일 (기본값: 2.0)
        r_cfg_scale: 참조 조건 스케일 (기본값: 1.0)
        e_cfg_scale: 감정 조건 스케일 (기본값: 1.0)
        emo: 감정 타입 (기본값: None, 'S2E' 사용)
        nfe: 추론 스텝 수 (기본값: 10)
        no_crop: 얼굴 크롭 비활성화 (기본값: False)
        seed: 랜덤 시드 (기본값: 25)
    
    Returns:
        생성된 비디오 파일
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 업로드된 파일 저장
        ref_path = os.path.join(temp_dir, ref_image.filename)
        audio_path = os.path.join(temp_dir, audio.filename)
        
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref_image.file, f)
        
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        # 결과 비디오 경로 설정
        import datetime
        call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        video_name = os.path.splitext(ref_image.filename)[0]
        audio_name = os.path.splitext(audio.filename)[0]
        
        res_video_path = os.path.join(
            agent.opt.res_dir,
            f"{call_time}-{video_name}-{audio_name}-nfe{nfe}-seed{seed}-acfg{a_cfg_scale}-ecfg{e_cfg_scale}-{emo if emo else 'S2E'}.mp4"
        )
        
        # 감정 설정 (None이면 'S2E' 사용)
        emotion = emo if emo else 'S2E'
        
        # 추론 실행
        result_path = agent.run_inference(
            res_video_path=res_video_path,
            ref_path=ref_path,
            audio_path=audio_path,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emotion,
            nfe=nfe,
            no_crop=no_crop,
            seed=seed,
            verbose=True
        )
        
        # 결과 파일이 존재하는지 확인
        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="비디오 생성에 실패했습니다.")
        
        # 비디오 파일 반환
        return FileResponse(
            path=result_path,
            media_type="video/mp4",
            filename=os.path.basename(result_path)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {str(e)}")
    
    finally:
        # 임시 파일 정리
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"임시 디렉토리 삭제 실패: {e}")


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": agent is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

