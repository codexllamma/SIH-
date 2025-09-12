import React from 'react';
import { alerts } from "../data/alerts"; 
import AlertCard from './AlertCard';

const AlertBox = () => {
  return (
    <div className="md:w-1/4 bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 flex flex-col h-full"> {/* Added h-full and flex flex-col */}
      <h2 className="text-xl font-semibold text-teal-300 mb-2">Alerts</h2>
      
      {/* Scrollable container for alerts */}
      <div className="flex-grow overflow-y-auto pr-2 custom-scrollbar"> {/* Added flex-grow and pr-2, custom-scrollbar */}
        <div className="flex flex-col gap-2"> {/* This div now exclusively manages the spacing of AlertCards */}
            {alerts.length > 0 ? (
            alerts.map((alert) => (
                <AlertCard
                key={alert.id}
                title={alert.title}
                message={alert.message}
                trainId={alert.trainId}
                />
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