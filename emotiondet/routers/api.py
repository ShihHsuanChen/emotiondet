from typing import List, Callable

from fastapi import APIRouter, Depends, Body

from emotiondet.model_manager import ModelManager

from ..schemas import PredictResult


def get_router(
        get_model_manager: Callable,
        **router_kwargs
        ):
    router = APIRouter(**router_kwargs)
    
    @router.post(
        '/get_emotion',
        response_model=PredictResult,
    )
    async def get_emotion(
            text: str = Body(),
            model_manager: ModelManager = Depends(get_model_manager),
            ):
        pred, prob = model_manager.single_inference(text)
        return PredictResult(pred=pred, prob=prob)
    return router
