from django.db import models
from django.contrib.auth.models import User 


class User(models.Model):
    user_id = models.IntegerField()
    user_name = models.CharField()
    gender = models.CharField()
    age = models.CharField()


class Item(models.Model):
    item_id = models.IntegerField()
    item_name = models.CharField


class Behavior(models.Model):
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    behave_time = models.TimeField()
    behave_type = models.CharField()
    score = models.IntegerField()
    

class Rec(models.Model):
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    rec_time = models.TimeField()
