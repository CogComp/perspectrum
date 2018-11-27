from django.contrib.auth.models import User
from django.contrib.auth import login
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect


@csrf_protect
def auth_login(request):
    """

    :param request
    :return:
    """
    username = request.POST['username']
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        user = User.objects.create_user(username)

    login(request=request, user=user)

    return HttpResponse(status=200) # The actual redirect happens front end, since ajax don't accept redirect response
