

import Header from "./components/Header";
import Map from "./components/Map";
import InfoBoxes from "./components/InfoBoxes";
import BridgeStatus from "./components/BridgeStatus";
import SuggestionsPanel from "./components/SuggestionsPanel";

function App() {
  return (
    <>
    <div className="flex flex-col space-y-3 bg-black p-4 h-full w-full">

      <Header section="Western" controller="Binod" status="Operational" />

      
      <div className="flex flex-row justify-between space-x-3 h-[80vh]">
        <Map />
        
      </div>


      <InfoBoxes />
    </div>
    <div>
        <SuggestionsPanel />
    </div>
    <div className="min-h-screen bg-black text-white flex justify-center items-center">
      <BridgeStatus />
    </div>
    </>
  );
}

export default App;
