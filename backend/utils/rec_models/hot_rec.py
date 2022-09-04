import pandas as pd
from .. import models


# ÈÈµãÍÆ¼ö
def hot_rec(rec_k=10):
    data = pd.DataFrame(models.Item.objects.values())
    rec_list = data.nlargest(rec_k, 'clicked_time')
    rec_ids = str(list(rec_list['item_id']))

    return rec_ids