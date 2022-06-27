from fastapi import APIRouter, HTTPException, status
from fastapi.param_functions import Depends
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from sqlalchemy.orm.session import Session
from DB.database import get_db
from DB import models
from DB.hash import Hash
from auth import oauth2

# from fastapi_jwt_auth import AuthJWT
# from fastapi_jwt_auth.exceptions import AuthJWTException
# from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse

app = FastAPI()

router = APIRouter(
    tags=['Authentication']
)

# class Settings(BaseModel):
#     authjwt_secret_key: str = "secret"


# @AuthJWT.load_config
# def get_config():
#     return Settings()

# @app.exception_handler(AuthJWTException)
# def authjwt_exception_handler(request: Request, exc: AuthJWTException):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"detail": exc.message}
#     )



@router.post('/token')
def get_token(request: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.DbUser).filter(models.DbUser.username == request.username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid credential")
    if not Hash.verify(user.password, request.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incorrect password")

    access_token = oauth2.create_access_token(data={'sub': user.username})

    return {
        'access_token': access_token,
        'token_type': 'bearer',
        'user_id': user.id,
        'username': user.username
    }       


# @router.post('/login')
# def login(request: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db), Authorize: AuthJWT = Depends(),):
# # def login(request: OAuth2PasswordRequestForm = Depends(), Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
#     user = db.query(models.DbUser).filter(models.DbUser.username == request.username).first()
#     if not user:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid credential")
#     if not Hash.verify(user.password, request.password):
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incorrect password")

#     # Use create_access_token() and create_refresh_token() to create our
#     # access and refresh tokens
#     access_token = Authorize.create_access_token(subject=user.username)
#     refresh_token = Authorize.create_refresh_token(subject=user.username)
#     return {"access_token": access_token, "refresh_token": refresh_token}
#     # return user.username





# #token: str = Depends(oauth2_scheme)
# @router.post('/refresh')
# def refresh(Authorize: AuthJWT = Depends()):
#     """
#     The jwt_refresh_token_required() function insures a valid refresh
#     token is present in the request before running any code below that function.
#     we can use the get_jwt_subject() function to get the subject of the refresh
#     token, and use the create_access_token() function again to make a new access token
#     """
#     Authorize.jwt_refresh_token_required()

#     current_user = Authorize.get_jwt_subject()
#     new_access_token = Authorize.create_access_token(subject=current_user)
#     return {"access_token": new_access_token}
