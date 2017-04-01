#include "NeuralNet.h"
#include <iostream>
NeuralNet::NeuralNet(const std::vector<unsigned> &topology){
	unsigned numLayers = topology.size();
	for(unsigned l= 0; l < numLayers; ++l){
		layers.push_back(Layer());
		unsigned numOutputs = l == numLayers - 1 ? 0: topology[l+1];

		for(unsigned n = 0; n <= topology[l]; ++n){
			layers.back().push_back(Neuron(numOutputs, n));
		}
		layers.back().back().setOutput(1.0);
	}
}

void NeuralNet::feedForward(const std::vector<double> &input){
	assert(input.size() == layers[0].size() - 1);

	for(unsigned i = 0; i < input.size(); ++i){
		layers[0][i].setOutput(input[i]);
	}

	for(unsigned l = 1; l < layers.size(); ++l){
		Layer &prevLayer = layers[l-1];
		for(unsigned n = 0; n < layers[l].size() - 1; ++n){
			layers[l][n].feedForward(prevLayer);
		}
	}
}

void NeuralNet::backProp(const std::vector<double> &target){
	Layer &outputLayer = layers.back();

	RMS = 0;

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
		double error = target[n] - outputLayer[n].getOutput();
		RMS += error * error;
	}

	RMS /= outputLayer.size() - 1;
	RMS = sqrt(RMS);

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
		outputLayer[n].calcOutputGradients(target[n]);
	}

	for(unsigned l = layers.size() - 2; l > 0; --l){
		for(unsigned n = 0; n < layers[l].size(); ++n){
			layers[l][n].calcHiddenGradients(layers[l+1]);
		}
	}

	for(unsigned l = layers.size() - 1; l > 0; --l){
		for(unsigned n = 0; n < layers[l].size() - 1; ++n){
			layers[l][n].updateInputsWeights(layers[l-1]);
		}
	}
}

void NeuralNet::getResults(std::vector<double> &result) const{
	result.clear();

	for(unsigned n = 0; n < layers.back().size() - 1; ++n){
		result.push_back(layers.back()[n].getOutput());
	}
}