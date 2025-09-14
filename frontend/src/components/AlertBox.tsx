import SuggestionsPanel from "./SuggestionsPanel";

const AlertBox = () => {
  return (
    <div className="md:w-1/4 bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 flex flex-col h-full"> 
      <h2 className="text-xl font-semibold text-teal-300 mb-2">Alerts</h2>
      

      <div className="flex-grow overflow-y-auto pr-2 custom-scrollbar"> 
        <div className="flex flex-col gap-2"> 
            {alerts.length > 0 ? (
            alerts.map((alert) => (
                <SuggestionsPanel />
            ))
            ) : (
            <p className="text-white font-semibold text-lg">No alerts currently</p>
            )}
        </div>
      </div>
    </div>
  );
};

export default AlertBox;
*/