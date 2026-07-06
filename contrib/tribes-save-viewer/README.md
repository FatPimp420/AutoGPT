# Tribes save schema + viewer

Notes on the full-game JSON save format used by [GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes)
(a Java framework/AI-competition codebase for *The Battle for Polytopia*), and a
single-file HTML viewer that plays back a sequence of these saves.

## Where it lives in the source

- Writer: `src/core/game/GameSaver.java` (package-private `writeTurnFile(...)`)
- Reader: `src/core/game/GameLoader.java`
- This is the **full game state** format, distinct from `src/core/game/LevelLoader.java`,
  which only loads the initial map layout (terrain/resources, no units/cities/turn state).

`GameSaver.writeTurnFile` is called once per tribe's turn. Each call writes one file to:

```
save/<seed>/<tick>_<activeTribeID>/game.json
```

So a full game is a *directory of many small JSON files*, one per (tick, active tribe)
pair, not one big file. The viewer below is built to load a batch of these files at once
and treat them as a timeline.

## JSON schema (top level)

```jsonc
{
  "board": { /* see below */ },
  "unit":  { /* map: actorId (string) -> unit info */ },
  "city":  { /* map: actorId (string) -> city info */ },
  "tribes": { /* map: actorId (string) -> tribe info */ },
  "seed": 123456789,
  "tick": 4,
  "gameIsOver": false,
  "activeTribeID": 0,
  "gameMode": 0
}
```

### `board`

All four board layers are `size x size` 2D arrays indexed `[x][y]`, where `size` is
`Board.getSize()` (implicit — it's just `terrain.length`). `-1` means "nothing here".

```jsonc
"board": {
  "terrain":  [[...]],   // Types.TERRAIN key per tile
  "resource": [[...]],   // Types.RESOURCE key per tile, or -1
  "unitID":   [[...]],   // actorId of the unit standing there, or -1
  "cityID":   [[...]],   // actorId of the city owning that tile, or -1
  "network":  [[...]],   // trade-network membership flag per tile
  "building": [[...]],   // Types.BUILDING key per tile, or -1
  "actorIDcounter": 42   // next free actor id, for continuing the game
}
```

`Types.TERRAIN`: `PLAIN=0, SHALLOW_WATER=1, DEEP_WATER=2, MOUNTAIN=3, VILLAGE=4, CITY=5, FOREST=6, FOG=7`

`Types.RESOURCE`: `FISH=0, FRUIT=1, ANIMAL=2, WHALES=3, ORE=5, CROPS=6, RUINS=7`

`Types.BUILDING`: `PORT=0, MINE=1, FORGE=2, FARM=3, WINDMILL=4, CUSTOMS_HOUSE=5, LUMBER_HUT=6,
SAWMILL=7, TEMPLE=8, WATER_TEMPLE=9, FOREST_TEMPLE=10, MOUNTAIN_TEMPLE=11, ALTAR_OF_PEACE=12,
EMPERORS_TOMB=13, EYE_OF_GOD=14, GATE_OF_POWER=15, GRAND_BAZAR=16, PARK_OF_FORTUNE=17,
TOWER_OF_WISDOM=18`

Note the board is a plain square grid (`Vector2d.neighborhood` uses an 8-direction/Chebyshev
neighborhood, not axial hex math). The GUI (`src/gui/GameView.java`) renders it with a 45°
isometric/diamond projection (`rotatePoint`), which is what makes it *look* hex-like — same
trick as the original Polytopia. The viewer below reproduces that projection.

### `unit` (map keyed by actor id string)

```jsonc
"unit": {
  "101": {
    "type": 0,            // Types.UNIT key
    "baseLandType": 0,     // only present for BOAT/SHIP/BATTLESHIP: the land unit it was built from
    "x": 3, "y": 5,
    "kill": 0,              // kill count
    "isVeteran": false,
    "cityID": 201,          // owning city's actor id
    "tribeId": 0,
    "currentHP": 10
  }
}
```

`Types.UNIT`: `WARRIOR=0, RIDER=1, DEFENDER=2, SWORDMAN=3, ARCHER=4, CATAPULT=5, KNIGHT=6,
MIND_BENDER=7, BOAT=8, SHIP=9, BATTLESHIP=10, SUPERUNIT=11`

### `city` (map keyed by actor id string)

```jsonc
"city": {
  "201": {
    "x": 3, "y": 5,
    "tribeID": 0,
    "population_need": 2,
    "bound": [...],           // city's bounding/work-radius tile list
    "level": 1,
    "isCapital": true,
    "population": 1,
    "production": 2,
    "hasWalls": false,
    "pointsWorth": 5,
    "buildings": [
      { "x": 3, "y": 5, "type": 8, "level": 1, "turnsToScore": 3 } // level/turnsToScore only for temples
    ],
    "units": [101, 102]        // actor ids of units garrisoned/owned by this city
  }
}
```

### `tribes` (map keyed by actor id string)

```jsonc
"tribes": {
  "0": {
    "citiesID": [201, 202],
    "capitalID": 201,
    "type": 0,                     // Types.TRIBE key
    "technology": {
      "researched": ["FISHING", "ORGANIZATION"],
      "everythingResearched": false
    },
    "star": 12,
    "winner": 0,                    // Types.WIN_STATUS key
    "score": 340,
    "obsGrid": [[...]],              // per-tile fog-of-war/observed flags for this tribe
    "connectedCities": [...],
    "monuments": { "12": 0 },         // BUILDING key -> MONUMENT_STATUS key
    "tribesMet": [1, 2],
    "extraUnits": [305],              // units not attached to any city
    "nKills": 4,
    "nPacifistCount": 0
  }
}
```

`Types.TRIBE`: `XIN_XI=0, IMPERIUS=1, BARDUR=2, OUMAJI=3, KICKOO=4, HOODRICK=5, LUXIDOOR=6,
VENGIR=7, ZEBASI=8, AI_MO=9, QUETZALI=10, YADAKK=11`

## Viewer

`index.html` is a self-contained (no build step, no external requests) page:

1. Click **Load save(s)** and select one or more `game.json` turn files (multi-select the
   `save/<seed>/*/game.json` files for one game). They're parsed, sorted by `tick`, and
   treated as animation frames.
2. The board renders as an isometric diamond grid on `<canvas>`: terrain color, city tiles
   tinted by owning tribe with their level, unit tokens colored by tribe with a type-letter
   abbreviation, small markers for resources/buildings.
3. Tap/click a tile to open an inspector panel with its terrain/resource/building/unit/city
   details.
4. Play/Pause, step, and a scrub slider move through the loaded turns; playback speed is
   adjustable.
5. Touch-friendly: drag to pan, pinch or wheel to zoom, big tap targets, safe-area padding
   for iOS.

Open it directly as a local file (`file://.../index.html`) or serve it with any static
file server — it makes no network calls.
