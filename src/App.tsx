
//import TrainSimulation from "./components/train-simulation";
//import Test from './components/test';
//import AlertCard from './components/AlertCard';

import AlertBox from "./components/AlertBox";
import Header from "./components/Header";
import Map from "./components/Map";

function App() {
  return (
    <div className="flex flex-col space-y-3 bg-black p-4 h-full w-full">
      <Header section="Western" controller="Binod" status="Operational" /> 
      <div className="flex flex-row justify-between space-x-3 h-[80vh]">
        <Map />
        <AlertBox /> 
      </div>
    </div>
  );
}

export default App