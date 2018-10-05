from django.contrib.auth.models import User
from django.contrib.auth import login
from django.http import HttpResponse


def auth_login(request):
    """

    :param request
    :return:
    """
    print(request.POST)
    username = request.POST['username']
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        user = User.objects.create_user(username)

    login(request=request, user=user)
    return HttpResponse(status=204)
