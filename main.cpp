#include "NeuralNet.h"
#include <vector>
#include <iostream>

int main(){
	srand(time(NULL));
	std::vector<unsigned> topology;
	topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);
	NeuralNet net(topology);

	std::vector<double> input1;
	input1.push_back(0.0);
	input1.push_back(0.0);
	std::vector<double> target1;
	target1.push_back(0.0);

	std::vector<double> input2;
	input2.push_back(0.0);
	input2.push_back(1.0);
	std::vector<double> target2;
	target2.push_back(1.0);

	std::vector<double> input3;
	input3.push_back(1.0);
	input3.push_back(0.0);
	std::vector<double> target3;
	target3.push_back(1.0);

	std::vector<double> input4;
	input4.push_back(1.0);
	input4.push_back(1.0);
	std::vector<double> target4;
	target4.push_back(0.0);

	std::vector<double> result;

	for(unsigned i = 0; i < 500; ++i){
		net.feedForward(input1);
		net.backProp(target1);
		net.feedForward(input2);
		net.backProp(target2);
		net.feedForward(input3);
		net.backProp(target3);
		net.feedForward(input4);
		net.backProp(target4);
	}
	std::cout << input1[0] << " " << input1[1] << " : ";
	net.feedForward(input1);
	net.getResults(result);
	std::cout << result.back() << std::endl;
	
	std::cout << input2[0] << " " << input2[1] << " : ";
	net.feedForward(input2);
	net.getResults(result);
	std::cout << result.back() << std::endl;

	std::cout << input3[0] << " " << input3[1] << " : ";
	net.feedForward(input3);
	net.getResults(result);
	std::cout << result.back() << std::endl;

	std::cout << input4[0] << " " << input4[1] << " : ";
	net.feedForward(input4);
	net.getResults(result);
	std::cout << result.back() << std::endl;
}