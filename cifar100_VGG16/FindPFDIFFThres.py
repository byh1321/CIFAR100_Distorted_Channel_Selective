import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--min', default=0, type=int, help='minimum threshold') 
parser.add_argument('--max', default=2000, type=int, help='maximum threshold') 
parser.add_argument('--interval', default=100, type=int, help='threshold interval') 
parser.add_argument('--input1', default='PFSUM_clean.txt', help='input txt name', metavar="FILE")
parser.add_argument('--input2', default='PFSUM_gau_02.txt', help='input2 txt name', metavar="FILE")

args = parser.parse_args()

max_correct = 0
max_thres = 1 

for thres in range(args.min,args.max+1,args.interval):
	correct=0
	f1 = open(args.input1,'r')
	f2 = open(args.input2,'r')
	while True:
		line = f1.readline()
		line2 = f2.readline()
		if not line: break
		correct = correct + (float(line[:-1]) < thres) + (float(line2[:-1]) > thres)	
	if correct > max_correct:
		max_correct = correct
		max_thres = thres
	f1.close()
	f2.close()
	if thres % (100 * args.interval) == 0:
		print("thres : ",thres)

#print("max_correct : ",max_correct, "(",max_correct/1281167/2*100,")")
print("max_correct : ",max_correct, "(",max_correct/50000/2*100,")")
print("max_thres : ",max_thres)

#73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
