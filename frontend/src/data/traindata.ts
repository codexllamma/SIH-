export const tracksData = [
  { id: "T1", points: [0, 80, 1000, 80] },   // line 1
  { id: "T2", points: [0, 200, 1000, 200] }, // line 2
  { id: "T3", points: [0, 320, 1000, 320] }, // line 3
  { id: "T4", points: [300, 80, 500, 200] }, // branch track
];

export const junctions = [
  { id: "J1", x: 300, y: 80, connectedTracks: ["T1", "T2", "T4"] },
  { id: "J2", x: 500, y: 200, connectedTracks: ["T2", "T4"] },
];

export const trains = [
  {
    id: "T1",
    label: "T1",
    colour: "red",
    x: 0,
    y: 80,   // aligned with T1
    speed: 2.5,
    delay: 0,

  },
  {
    id: "T2",
    label: "T2",
    colour: "blue",
    x: 0,
    y: 200,  // aligned with T2
    speed: 3.0,
    delay: 10,
    
  },
  {
    id: "T3",
    label: "T3",
    colour: "green",
    x: 0,
    y: 320,  // aligned with T3
    speed: 2.0,
    delay: 5,
    
  },
  {
    id: "T4",
    label: "T4",
    colour: "orange",
    x: 0,
    y: 80,   // reusing T1 track for now
    speed: 1.8,
    delay: 0,

  }
];

export const suggestions = [
  {
    trainId: "T2",
    suggestionId: "S1",
    type: "reroute",
    message: "Divert Shatabdi via Gwalior to save 8 minutes.",
    impact: -8
  },
  {
    trainId: "T3",
    suggestionId: "S2",
    type: "hold",
    message: "Hold Duronto for 5 minutes at Surat to avoid congestion.",
    impact: 5
  },
  {
    trainId: "T1",
    suggestionId: "S3",
    type: "priority",
    message: "Give Rajdhani Express priority crossing near Kanpur.",
    impact: -12
  }
];

export const delays = [
  { trainId: "T1", station: "Kanpur", scheduled: "12:30", actual: "12:45", delay: 15 },
  { trainId: "T2", station: "Jhansi", scheduled: "14:10", actual: "14:25", delay: 15 },
  { trainId: "T3", station: "Vadodara", scheduled: "16:00", actual: "16:05", delay: 5 }
];

export const platforms = [
  { id: "APF1", label: "A/PF1", colour: "lightblue", x: 50, y: 110 },
  { id: "APF2", label: "A/PF2", colour: "lightblue", x: 50, y: 240 },
  { id: "BPF1", label: "B/PF1", colour: "lavender", x: 720, y: 110 },
  { id: "BPF2", label: "B/PF2", colour: "lavender", x: 720, y: 240 },
];

export const signals = [
  { id: 1, x: 300, y: 80 },
  { id: 2, x: 500, y: 200 },
  { id: 3, x: 150, y: 80 },
  { id: 4, x: 150, y: 200 },
  { id: 5, x: 820, y: 80 },
  { id: 6, x: 820, y: 200 },
];


export const trainRoutes: { [trainId: string]: string[] } = {
  T1: ["T1", "T4", "T2"], 
  T2: ["T2"],
  T3: ["T3"],
  T4: ["T1"]
};
