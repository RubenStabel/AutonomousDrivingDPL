nn(ad_baseline_speed_net,[Img, speed],X,[0,1,2,3]) :: action(Img,X).

autonomous_driving_baseline_speed(Img, speed, X) :- action(Img,X).