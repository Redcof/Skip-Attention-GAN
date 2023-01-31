# Class List

```
["NORMAL0", "NORMAL1", "HUMAN",
"UNKNOWN", "UN", "CP", "MD", "GA", "KK", "SS", "KC", "WB", "LW", "CK", "CL"]
```

## Index Map:

0. "NORMAL0",
0. "NORMAL1",
0. "HUMAN",
0. "UNKNOWN",
0. "UN",
0. "CP",
0. "MD",
0. "GA",
0. "KK",
0. "SS",
0. "KC",
0. "WB",
0. "LW",
0. "CK",
0. "CL"

## Acronyms

```
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

## Special Classes

```
HUMAN="HUMAN", 
NORMAL0=General patch with no object, 
NORMAL1=General patch with no object
```

# ATZ Patch dataset file format:

`atz_patch_dataset__c_w_n.csv`

```
where c = class index in 'Class List; form which the bbox consedered
      w = patch size
      n = no of image patches per image with patch size w
```

> **Note**: 'HUMAN' bbox are always ignored

Below files contains the list of file to be used for training and testing

* `atz_dataset_test_ablation.csv`
* `atz_dataset_train_ablation.csv`

*For `SAGAN`:*
For `train` dataset Please make sure to specify file names in which no object are found
For `test` dataset no restriction is imposed

## Additional formal details

- ATZ images are `w x h`=`335 x 880` pixels
- If we split `64 x 64` patches(w=64)
- with `20%` overlap in both side we get `n=119` patches per image.
- This dataset contains only classes starting from `c=12` in `Class List("LW", "CK", and "CL")`
- So the filename is: `atz_patch_dataset__12_64_119.csv`
- `atz_patch_dataset__3_128_36_v2_10%_30_99%_nomulti.csv` now for this type file we perform few additional operation
  1. We check the goodness(mostly black) of a patch by performing a threshold color check with a percentage
  2. the current color value is 30, with 99% threshold.
  3. This implies if a patch having 99% pixels with color value 30 or less will be discarded from the dataset
  4. 

# CSV Headers

1. `image` = source image
2. `patch_id` = patch index starts with 0
3. `label` = class index in the above list
4. `label_txt` = label text
5. `anomaly_size`= pixel count of bbox encountered in the patch
6. `x1x2y1y2` = global location of the patch in the image
7. `threat_present`= threat available or not
8. `front_back`= front side or back side
9. `global_x1y1x2y2`= if threat available, bbox w.r.t main image
10. `relative_x1y1x2y2`= if threat available, relative bbox w.r.t the patch image
11. `subject_gender`= subject gender(F/M)
12. `subject_id`=subject id(M1, F1, F2, M2,etc.)



