"""webapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from webapp import views
from webapp import auth

urlpatterns = [
    path('', views.render_login_page, name="render_login"),
    path('admin/', admin.site.urls),
    path('main/', views.main_page, name="main_page"),
    path('getJson/', views.get_json),
    path('claims/', views.vis_claims),
    path('perspectives/<slug:claim_id>', views.vis_persps, name="perspectives"),
    path('claim_neg_anno/<slug:claim_id>', views.vis_neg_anno, name="claim_neg_anno"),
    path('claim_relation/<slug:claim_id>', views.vis_relation, name="claim_relation"),
    path('api/submit_rel_anno/', views.submit_rel_anno, name="submit_rel_anno"),
    path('api/auth_login/', auth.auth_login, name="auth_login"),
]
