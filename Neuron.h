#include <vector>
#include <cstdlib>
#include <cmath>

struct Connection
{
	double weight;
	double deltaWeight;	
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned _index);
	void feedForward(const Layer &prevLayer);
	void setOutput(double val){output = val;}
	double getOutput() const{return output;}
	void calcOutputGradients(double target);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputsWeights(Layer &prevLayer);
	
private:
	static double eta;
	static double alpha;
	static double activationFunction(double x);
	static double activationFunctionDer(double x);
	double output;
	double gradient;
	std::vector<Connection> weights;
	unsigned index;
};