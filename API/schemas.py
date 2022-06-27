from typing import List
from pydantic import BaseModel

#this data type comes from user
class UserBase(BaseModel):
  username: str
  # email: str
  password: str
  # first_name: str
  # last_name: str
  # phone: str

#this data type will be send back to user
class UserDisplay(BaseModel):
  username: str
  # email: str
  #this config class convert data from db to our format
  #our format is above format(username, email)
  #without config class, we would get error
  class Config(): 
    orm_mode = True