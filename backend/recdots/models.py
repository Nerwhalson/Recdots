from django.db import models

# Create your models here.


class User(models.Model):
    user_id = models.IntegerField()
    user_name = models.CharField(max_length=255)
    gender = models.CharField(max_length=255)
    age = models.IntegerField()


class Item(models.Model):
    item_id = models.IntegerField()
    item_name = models.CharField(max_length=255)


class Behavior(models.Model):
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    behave_time = models.TimeField()
    behave_type = models.CharField(max_length=255)
    score = models.IntegerField()
    

class Rec(models.Model):
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    rec_time = models.TimeField()
