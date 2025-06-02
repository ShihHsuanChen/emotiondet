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
        title='Emotiondet API',
        description='Inference API for emotion detection using the fintuned Deberta model.',
        lifespan=lifespan,
    )

    # dependencies
    def get_model_manager():
        yield model_manager

    # routers.ui.include(app) # usage 1
    app.mount('/static', routers.ui.staticfiles) # usage 2
    app.include_router(routers.ui.get_router('/static'), tags=['api']) # usage 2
    app.include_router(routers.api.get_router(get_model_manager), prefix='/api', tags=['api'])
    return app
