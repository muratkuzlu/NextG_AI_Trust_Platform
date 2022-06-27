#import fastAPI from fast api
from fastapi import FastAPI, Request

#import DB dependencies from local folders/files
from DB import models
# from project import models
from DB.database import engine

#import userAPI from router
# from router import user

# import auth router from auth
# from auth import authentication
from auth import authentication
from DB import user_API

# import fast-api cors
from fastapi.middleware.cors import CORSMiddleware

#decleration fast api instance
app = FastAPI()
app.include_router(authentication.router)
app.include_router(user_API.router)

origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#api end point
@app.get(
    '/hello',
     tags=['Test'],
     summary='This is a simple API for small test'
    # description="Create an item with all the information, name, description, price, tax and a set of unique tags"
    )
def index():
    """
    Additional description 
    """
    return {'message': 'Hello world! This is the simple test api'}


#create DB
models.Base.metadata.create_all(engine)

