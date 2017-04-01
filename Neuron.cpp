#include "Neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.3;

Neuron::Neuron(unsigned numOutputs, unsigned _index){
	index = _index;
	for(unsigned c = 0; c < numOutputs; ++c){
		weights.push_back((Connection()));
		weights.back().weight = ((double)rand()/(double) RAND_MAX)*2 - 1;
		weights.back().deltaWeight = 0;
	}
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;

	for(unsigned n = 0; n < prevLayer.size(); ++n){
		sum += prevLayer[n].getOutput() * prevLayer[n].weights[index].weight;
	}

	output = Neuron::activationFunction(sum);
}

void Neuron::calcOutputGradients(double target){
	double delta = target - output;
	gradient = delta * Neuron::activationFunctionDer(output);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
	double sum = 0.0;

	for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
		sum += weights[n].weight * nextLayer[n].gradient;
	}

	gradient = sum * Neuron::activationFunctionDer(output);
}

void Neuron::updateInputsWeights(Layer &prevLayer){
	for(unsigned n = 0; n < prevLayer.size(); ++n){
		double oldDW = prevLayer[n].weights[index].deltaWeight;
		double newDW = eta*prevLayer[n].getOutput()*gradient + alpha*oldDW;
		prevLayer[n].weights[index].deltaWeight = newDW;
		prevLayer[n].weights[index].weight += newDW;
	}
}

double Neuron::activationFunction(double x){
	return tanh(x);
	//return 1/(1+exp(-x));
}

double Neuron::activationFunctionDer(double x){
	return 1.0-x*x;
	//return Neuron::activationFunction(x)*(1 - Neuron::activationFunction(x));
}