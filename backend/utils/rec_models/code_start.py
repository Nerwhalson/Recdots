import pandas as pd
from .. import models


# ÀäÆô¶¯ÍÆ¼ö
def init_rec(user_id, rec_k=10):
    try:
        user = pd.DataFrame(models.User.objects.filter(user_id=user_id).values())
    except:
        return False

    user_type = user['user_type'][0]
    item_data = pd.DataFrame(models.Item.objects.values())
    item_data['item_type'].apply(lambda x: x.split, args=('|',))
    item_data = item_data[user_type in item_data['item_type']]
    item_data = item_data.sort_values(by='clicked_time', ascending = False)
    item_rec = item_data[:rec_k]
    rec_ids = str(list(item_rec['item_id']))

    return rec_ids
