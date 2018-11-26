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
    # commmon urls
    path('', views.render_login_page, name="render_login"),
    path('logout/', views.logout_request, name="logout"),
    path('contact/', views.render_contact, name="contact"),
    path('success/', views.successView, name='success'),
    path('admin/', admin.site.urls),
    path('main/', views.main_page, name="main_page"),
    path('get_json/', views.get_json),
    path('claims/', views.vis_claims),
    path('perspectives/<slug:claim_id>', views.vis_persps, name="perspectives"),
    path('vis_spectrum/<slug:claim_id>', views.vis_spectrum, name="vis_spectrum"),

    # step 1
    path('step1/task_list/', views.render_list_page, name="task_list"),
    path('step1/instructions/', views.render_instructions, name="instructions"),
    path('claim_neg_anno/<slug:claim_id>', views.vis_neg_anno, name="claim_neg_anno"),
    path('claim_relation/<slug:claim_id>', views.vis_relation, name="claim_relation"),
    path('normalize_persp/<slug:claim_id>', views.vis_normalize_persp, name="normalize_claim"),
    path('api/submit_rel_anno/', views.submit_rel_anno, name="submit_rel_anno"),
    path('api/auth_login/', auth.auth_login, name="auth_login"),
    path('api/submit_instr/', views.submit_instr, name="submit_instr"),

    # step 2
    path('step2/perspective_equivalence/<slug:claim_id>', views.vis_persp_equivalence, name="perspective_equivalence"),
    path('step2/instructions/', views.render_step2_instructions, name="step2_instructions"),
    path('step2/task_list/', views.render_step2_task_list, name="step2_task_list"),
    path('step2/api/submit_equivalence_annotation', views.submit_equivalence_annotation, name="submit_equivalence_annotation"),
    path('step2/api/submit_instr/', views.step2_submit_instr, name="step2_submit_instr"),

    # step 3
    path('step3/verify_evidence/<slug:batch_id>', views.render_evidence_verification, name="verify_evidence"),
    path('step3/instructions/', views.render_step3_instructions, name="step3_instructions"),
    path('step3/task_list/', views.render_step3_task_list, name="step3_task_list"),
    path('step3/api/submit_annotation', views.submit_evidence_annotation,
         name="submit_evidence_annotation"),
    path('step3/api/submit_instr/', views.step3_submit_instr, name="step3_submit_instr"),
]
