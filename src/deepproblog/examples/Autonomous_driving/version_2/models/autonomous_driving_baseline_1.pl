nn(perc_net_version_2_baseline_1,[Img,Speed],X,[0,1,2,3]) :: action(Img,Speed,X).

autonomous_driving_baseline(Img, Speed, X) :- action(Img,Speed,X).