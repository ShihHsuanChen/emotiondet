from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model_manager import ModelManager, check_requirements
from .schemas import PredictResult
from .settings import AppSettings
from . import routers


def create_app(settings: AppSettings):
    model_manager: ModelManager = ModelManager(
        settings.model_dir,
        max_length=settings.max_length,
        batch_size=settings.batch_size,
        device=settings.device,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print('Checking dependencies ... ', end='')
        try:
            check_requirements()
        except:
            print('Fail')
            raise
        else:
            print('Ok')

        print('Load model ... ', end='')
        try:
            model_manager.load_model()
        except:
            print('Fail')
            raise
        else:
            print('Done')
        yield

    app = FastAPI(
        prog='emotiondet',
        desc='Inference API for emotion detection using the fintuned Deberta model.',
        lifespan=lifespan,
    )

    # dependencies
    def get_model_manager():
        yield model_manager

    app.mount('/static', routers.ui.staticfiles)
    routers.ui.include(app)
    app.include_router(routers.api.get_router(get_model_manager), prefix='/api', tags=['api'])
    return app
