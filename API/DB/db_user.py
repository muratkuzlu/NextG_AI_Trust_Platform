from DB.hash import Hash
from sqlalchemy.orm.session import Session
from schemas import UserBase
from DB.models import DbUser
from fastapi import HTTPException, status


#this is a db query file for CRUD 
#Creating data to DB
def create_user(db: Session, request: UserBase):
  new_user = DbUser(
    username = request.username,
    # email = request.email,
    password = Hash.bcrypt(request.password)
    # first_name = request.first_name,
    # last_name = request.last_name
  )
  db.add(new_user)
  db.commit()
  db.refresh(new_user)
  return new_user

#read all user from db
def get_all_users(db: Session):
  return db.query(DbUser).all()