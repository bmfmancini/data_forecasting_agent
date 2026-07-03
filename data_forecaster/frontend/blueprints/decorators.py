
from functools import wraps
from flask import current_app, flash, redirect, request, url_for
from flask_login import current_user

def password_change_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login', next=request.url))
        
        if getattr(current_user, 'must_change_password', False):
            if request.endpoint and request.endpoint != 'auth.change_password':
                flash('You must change your password before you can continue.', 'warning')
                return redirect(url_for('auth.change_password'))
        
        return f(*args, **kwargs)
    return decorated_function
