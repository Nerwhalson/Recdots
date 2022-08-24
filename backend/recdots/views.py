from django.shortcuts import render
from django.http import HttpResponse
from django.utils.timezone import now
from . import models

# Create your views here.

def report_item(request):
    if request.method == 'GET':
        return render(request, 'register.html')
    elif request.method == "POST":
        item_list = request.POST.get("ItemList", None)
        for item in item_list:
            models.Item.objects.create(item_id=item['ItemId'], item_name=item["ItemName"])
        return HttpResponse("add item finish")


def report_user(request):
    return HttpResponse("add user finish")


def report_behavior(request):
    return HttpResponse("add behavior finish")


def recommend(request):
    return HttpResponse("return recommendation")
