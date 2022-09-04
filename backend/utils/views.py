from django.shortcuts import render
from django.http import HttpResponse
from django.utils.timezone import now

from django.db import connection
from . import models
from .rec_models import MF

# Create your views here.

def report_item(request):
    if request.method == "POST":
        item_list = request.POST.getlist("ItemList", [])
        for item in item_list:
            models.Item.objects.create(item_id=item['ItemId'], item_name=item["ItemName"], create_date=now())
        return HttpResponse("add item finish")


def report_user(request):
    if request.method == "POST":
        user_list = request.POST.getlist("UserList", [])
        for user in user_list:
            models.Item.objects.create(
                user_id=user['ItemId'], 
                user_name=user["ItemName"], 
                age=user["Age"], 
                gender=user["Gender"], 
                create_date=now())
        return HttpResponse("add user finish")


def report_behavior(request):
    if request.method == "POST":
        behave_list = request.POST.getlist("BehaviorList", [])
        for behave in behave_list:
            UserId =behave["UserId"]
            ItemId =behave["ItemId"]
            BehaveTime =behave["BehaviorTimestamp"]
            BehaveType =behave["BehaviorType"]
            Score =behave["Score"]
            models.Behavior.objects.create(user_id=UserId, item_id=ItemId, behave_time=BehaveTime, behave_type=BehaveType, score=Score)
        return HttpResponse("add behavior finish")


def recommend(request):
    if request.method == "POST":
        user_list = request.POST.getlist("UserList", [])
        for user in user_list:
            user_obj = models.Rec.objects.filter(user_id=user)
            behave_obj = models.Behavior.objects.filter(user_id=user)
            models.Behavior.objects.create(user_id=user, item_id=ItemId, behave_time=now(), behave_type="Rec", score=None)
            
    

    return HttpResponse("return recommendation")


def offline_training(request):
    MF.offline_rec()

    return HttpResponse(MF.REC_PATH)
