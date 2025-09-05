export const tracksData = [
  { id: "T1", points: [0, 80, 1000, 80] }, // line 1

  { id: "T2", points: [0, 200, 1000, 200] }, // line 2

  { id: "T3", points: [0, 320, 1000, 320] }, // line 3

  { id: "T4", points: [300, 80, 500, 200] }, // line that joins
];

// junctions (only these have signals). connectedTracks lists IDs of tracks meeting here
export const junctions = [
  { id: "J1", x: 300, y: 80, connectedTracks: ["T1", "T2"] },
  { id: "J2", x: 500, y: 200, connectedTracks: ["T1", "T2"] },
];

// trains: each train is bound to a track (by trackId) and has a progress (0..1) and speed (fraction per second)
export const trains = [
  { id: "T1", label: "T1", colour: "aqua", x: 0, y: 80 },
  { id: "T2", label: "T2", colour: "white", x: 0, y: 200 },
  { id: "T3", label: "T3", colour: "yellow", x: 0, y: 320 },
];

export const platforms = [
  { id: "APF1", label: "A/PF1", colour: "lightblue", x: 50, y: 110 },
  { id: "APF1", label: "A/PF2", colour: "lightblue", x: 50, y: 240 },
  { id: "BPF1", label: "B/PF1", colour: "lavender", x: 720, y: 110 },
  { id: "BPF1", label: "B/PF2", colour: "lavender", x: 720, y: 240 },
];

// data/signals.ts
export const signals = [
  { id: 1, x: 300, y: 80 },
  { id: 2, x: 500, y: 200 },
  { id: 3, x: 150, y: 80 },
  { id: 4, x: 150, y: 200 },
  { id: 5, x: 820, y: 80 },
  { id: 6, x: 820, y: 200 },
];

/*
const pastelColors = {
  pink: "#FADADD",      // 
  peach: "#FFDAB9",     // pastel peach
  mint: "#B9FBC0",      // mint green
  babyBlue: "#A7C7E7",  // baby blue
  lavender: "#E6E6FA",  // lavender
  lemon: "#FFFACD",     // pastel yellow
  coral: "#F6D7B0",     // soft coral
  lilac: "#D8BFD8",     // lilac purple
};
*/
