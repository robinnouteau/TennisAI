import datetime
from datetime import timezone


def get_last_ok_sessions(sessions):
    last_date = datetime.datetime(2000, 1, 1, tzinfo=timezone.utc)
    last_session = None
    for session in sessions:
        if session['status'] == 'OK':
            date = datetime.datetime.fromisoformat(session['created_at'].replace('Z', '+00:00'))
            date = date.replace(tzinfo=timezone.utc).astimezone(tz=None)
            if date > last_date:
                last_session = session
                last_date = date
    return last_session, last_date
