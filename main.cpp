#include "NeuralNet.h"
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


class Trainer{
public:
	std::ifstream file;
	Trainer(std::string path){
		file.open(path.c_str());
		std::string line;
		std::getline(file, line);
		std::stringstream ss(line);
		unsigned x;
		while(ss >> x)
			topology.push_back(x);
	}

	bool isEof(){
		return file.eof();
	}

	void getNextRow(std::vector<double> & input){
		input.clear();
		std::string line;
		std::getline(file, line);
		std::stringstream ss(line);
		unsigned x;
		while(ss >> x)
			input.push_back(x);
	}

	void getTopology(std::vector<unsigned> & input){
		input.clear();
		for(unsigned i = 0; i < topology.size(); ++i){
			input.push_back(topology[i]);
		}
	}

	std::vector<unsigned> topology;
};

int main(){
	srand(time(NULL));

	Trainer trainer("tmp/not.txt");
	std::vector<unsigned> topology;
	trainer.getTopology(topology);
	std::vector<double> input;
	std::vector<double> target;
	std::vector<double> result;
	NeuralNet net(topology);
	
	unsigned trainingCount = 0;
	while(!trainer.isEof()){
		++trainingCount;
		trainer.getNextRow(input);
		trainer.getNextRow(target);
		net.feedForward(input);
		net.getResults(result);
		net.backProp(target);

		std::cout << "Pass: " << trainingCount << std::endl;

		std::cout << "Inputs: ";
		for(unsigned i = 0; i < input.size(); ++i)
			std::cout << input[i] << " ";
		std::cout << std::endl;

		std::cout << "Results: ";
		for(unsigned i = 0; i < result.size(); ++i)
			std::cout << result[i] << " ";
		std::cout << std::endl;

		std::cout << "Targets: ";
		for(unsigned i = 0; i < target.size(); ++i)
			std::cout << target[i] << " ";
		std::cout << std::endl << std::endl;
	}

	/*

	std::vector<double> customInput;
	customInput.push_back(1);
	customInput.push_back(1);
	customInput.push_back(1);
	customInput.push_back(1);
	customInput.push_back(0);
	std::vector<double> customResult;
	net.feedForward(customInput);
	net.getResults(customResult);

	std::cout << "[1,1,1,1,0] -> ";

	for(unsigned i = 0; i < customResult.size(); ++i)
		std::cout << customResult[i] << " ";
	std::cout << std::endl;

	customInput.clear();
	customInput.push_back(0);
	customInput.push_back(0);
	customInput.push_back(0);
	customInput.push_back(0);
	customInput.push_back(1);

	net.feedForward(customInput);
	net.getResults(customResult);

	std::cout << "[0,0,0,0,1] -> ";

	for(unsigned i = 0; i < customResult.size(); ++i)
		std::cout << customResult[i] << " ";
	std::cout << std::endl;

	*/
}