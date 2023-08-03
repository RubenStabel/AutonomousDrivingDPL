nn(perc_net_version_5_baseline_0,[Img,MNIST,Speed,Player_y,Intersection_y],X,[0,1,2,3]) :: action(Img,MNIST,Speed,Player_y,Intersection_y,X).

autonomous_driving_baseline(Img, MNIST, Speed, Player_y, Intersection_y, X) :- action(Img,MNIST,Speed,Player_y,Intersection_y,X).