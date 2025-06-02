import os

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


staticfiles = StaticFiles(directory=os.path.join(os.path.dirname(__file__), '../static'))


def include(app: FastAPI):
    app.mount('/static', staticfiles)

    @app.get('/', name='root')
    async def root(request: Request):
        return RedirectResponse('/static/index.html')


def get_router(static_prefix: str = '/static', **router_kwargs):
    router = APIRouter(**router_kwargs)

    @router.get('/', name='root')
    async def root(request: Request):
        return RedirectResponse(static_prefix + '/index.html')
    return router
