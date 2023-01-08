# class_list = 
```
["NORMAL0", "NORMAL1", "HUMAN","UNKNOWN", "UN", "CP", "MD", "GA", "KK", "SS", "KC", "WB", "LW", "CK", "CL"]

KK="Kitchen Knife", 
GA="Gun", 
MD="Metal Dagger",
SS="Scissors", 
WB="Water Bottle", 
CK="Ceramic Knife",
CP="Cell Phone", 
KC="Key Chain", 
LW="Leather Wallet",
CL="Cigarette Lighter", 
UN="UNKNOWN", UNKNOWN="UNKNOWN",
```

```
HUMAN="HUMAN", 
NORMAL0=General patch with no object, 
NORMAL1=General patch with no object
```
File format: atz_patch_dataset__c_w_n.csv
where c = is class index from which the bbox consedered
      w = patch size
      n = no of image patches per image with patch size w


Columns:
image = source image
patch_id = patch index starts with 0
label = class index in the above list
label_txt = label text
anomaly_size= pixel count of bbox encountered in the patch
x1x2y1y2 = global location of the patch in the image
