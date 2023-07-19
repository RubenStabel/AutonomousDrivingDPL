nn(perc_net_version_2_baseline_2,[Img,MNIST,Speed],X,[0,1,2,3]) :: action(Img,MNIST,Speed,X).

autonomous_driving_baseline(Img, MNIST, Speed, X) :- action(Img,MNIST,Speed,X).