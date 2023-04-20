nn(ad_baseline_net,[Img],X,[0,1,2,3,4,5,6,7,8]) :: action(Img,X).

autonomous_driving_baseline(Img,X) :- action(Img,X).