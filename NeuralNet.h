#include <vector>
#include <cassert>

class Neuron;

typedef std::vector<Neuron> Layer;

#include "Neuron.h"

class NeuralNet{
public:
	NeuralNet(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &input);
	void backProp(const std::vector<double> &target);
	void getResults(std::vector<double> &result) const;

private:
	std::vector<Layer> layers;
	double RMS;
};