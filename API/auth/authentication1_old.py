from fastapi import APIRouter


router = APIRouter(
    prefix='/auth',
    tags=['auth']
)

# email verification
@router.get('/email-verify/')
def email_verify():
    return 'email verify API'

# login API
@router.post('/login/')
def login():
    return 'login API'

# logout API
@router.post('/logout/')
def logout():
    return 'logout'

# password reset complate
@router.get('/password-reset/{uidb64}/{token}/')
def password_reset():
    return 'Password reset'

# register
@router.post('/register/')
def register():
    return 'Password reset'

# request reset email
@router.post('/request-reset-email/')
def request_reset_email():
    return 'Request reset email'