from typing import List
from schemas import UserBase, UserDisplay
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from DB.database import get_db
from DB import db_user
# from auth.oauth2 import get_current_user

router = APIRouter(
  prefix='/user',
  tags=['user']
)

#Create user
@router.post('/', response_model=UserDisplay)#use UserDisplay class as response for our provided data type from schemas
def create_user(request: UserBase, db: Session = Depends(get_db)):
  return db_user.create_user(db, request)

# #Read all users
@router.get('/', response_model=List[UserDisplay])
def get_all_users(db: Session = Depends(get_db)):
# def get_all_users(db: Session = Depends(get_db), current_user: UserBase = Depends(get_current_user)):
  return db_user.get_all_users(db)

# #read one user
# @router.get('/{id}', response_model=UserDisplay)
# def get_user(id: int, db: Session = Depends(get_db)):
#   return db_user.get_user(db, id)

# #Update user
# @router.post('/{id}/update')
# def update_user(id: int, request: UserBase, db: Session = Depends(get_db)):
#   return db_user.update_user(db, id, request)


# #Delete user
# @router.get('/delete/{id}')
# def delete(id: int, db: Session = Depends(get_db)):
#   return db_user.delete_user(db, id)