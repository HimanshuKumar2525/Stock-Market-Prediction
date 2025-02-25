from . import views
from django.urls import path 

urlpatterns = [
    
    path('',views.home,name="home" ),
    path('signup',views.signup,name="signup" ),
    path('login',views.login_view,name="login" ),
    path('logout', views.logout_view, name='logout'),
    path('activate/<uidb64>/<token>',views.ActivateAccountView.as_view(),name='activate'),
    path('request-reset-email',views.RequestResetEmailView.as_view(),name='request-reset-email'),
    path('set-new-password<uidb64>/<token>',views.SetNewPasswordView.as_view(),name='set-new-password'),
    path('prediction',views.prediction,name="prediction" ),
    path('about',views.about,name="about" ),
    path('contact',views.contact,name="contact" ),
    
]