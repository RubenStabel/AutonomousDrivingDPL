nn(ad_baseline_net,[Img],X,[0,1,2]) :: action(Img,X).

autonomous_driving_baseline(Img,X) :- action(Img,X).