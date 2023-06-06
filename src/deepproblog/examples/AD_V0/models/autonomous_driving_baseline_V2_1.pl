nn(ad_baseline_speed_net,[Img,Speed],X,[0,1,2,3]) :: action(Img,Speed,X).

autonomous_driving_baseline_speed(Img, Speed, X) :- action(Img,Speed,X).