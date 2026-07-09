package core.game;

import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.CityAction;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.ResourceGathering;
import core.actions.cityactions.Spawn;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.EndTurn;
import core.actions.tribeactions.ResearchTech;
import core.actions.tribeactions.TribeAction;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Convert;
import core.actions.unitactions.Move;
import core.actions.unitactions.UnitAction;
import core.actors.Actor;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import players.Agent;
import players.DoNothingAgent;
import players.RandomAgent;
import players.SimpleAgent;
import players.osla.OneStepLookAheadAgent;
import players.osla.OSLAParams;
import players.mcts.MCTSPlayer;
import players.mcts.MCTSParams;
import utils.ElapsedCpuTimer;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

/**
 * Step-wise driver of a Tribes game for external (e.g. Python RL) control.
 *
 * Lives in core.game to reach the package-private turn lifecycle of GameState
 * (init, initTurn, computePlayerActions, isTurnEnding, incTick, gameOver).
 *
 * GameState.advance() rotates the active tribe on EndTurn but never increments
 * the game tick (Game.tick() owns that in the normal loop), so this runner
 * increments it whenever the rotation wraps back to a lower tribe index.
 */
public class RLGameRunner {

    // Number of ints per action descriptor in getLegalActionData().
    public static final int ACTION_FIELDS = 8;
    // Number of channels per tile in getObservation().
    public static final int OBS_CHANNELS = 13;
    // Number of entries in getScalars().
    public static final int SCALAR_FIELDS = 10;

    private static final int MAX_MICRO_STEPS = 20000; // hard safety cap per game

    private GameState gs;
    private Agent[] bots; // null entry = externally controlled player
    private ArrayList<Action> legal = new ArrayList<>();
    private int microSteps;
    // When false, bot seats do not auto-play inside reset()/step(); the caller
    // drives them one action at a time via stepBot() so every action can be
    // observed (used for recorded exhibition games).
    private boolean autoPlayBots = true;
    private int mapSizeOverride = 0;  // 0 = default per player count

    public void setAutoPlayBots(boolean auto) { this.autoPlayBots = auto; }
    public void setMapSize(int s) { this.mapSizeOverride = s; }

    /** True if the active seat is a bot (only meaningful with autoplay off). */
    public boolean activeSeatIsBot() {
        return !gs.isGameOver() && bots[gs.getActiveTribeID()] != null;
    }

    /** Plays ONE action chosen by the active seat's bot agent.
     *  Returns the played action's ACTION_FIELDS descriptor. */
    public int[] stepBot() {
        int[] desc = new int[ACTION_FIELDS];
        java.util.Arrays.fill(desc, -1);
        if (gs.isGameOver() || bots[gs.getActiveTribeID()] == null) return desc;
        int botId = gs.getActiveTribeID();
        ElapsedCpuTimer ect = new ElapsedCpuTimer();
        ect.setMaxTimeMillis(1000);
        gs.computePlayerActions(gs.getActiveTribe());
        Action a = gs.getAllAvailableActions().isEmpty() ? null
                : bots[botId].act(gs.copy(botId), ect);
        if (a == null) a = new EndTurn(botId);
        encodeAction(a, desc, 0);
        applyAction(a);
        refreshLegal();
        return desc;
    }

    /**
     * Starts a fresh game on a generated level.
     *
     * @param levelSeed seed for the level generator
     * @param gameSeed  seed for in-game randomness
     * @param tribeNames Types.TRIBE names, one per player (e.g. {"XIN_XI", "IMPERIUS"})
     * @param gameMode  "CAPITALS" or "SCORE"
     * @param botTypes  per player: "" for external control, or "random"/"simple"/"donothing"
     */
    public void reset(long levelSeed, long gameSeed, String[] tribeNames, String gameMode, String[] botTypes) {
        Types.GAME_MODE mode = gameMode.equalsIgnoreCase("CAPITALS") ?
                Types.GAME_MODE.CAPITALS : Types.GAME_MODE.SCORE;

        int nPlayers = tribeNames.length;
        Types.TRIBE[] tribes = new Types.TRIBE[nPlayers];
        for (int i = 0; i < nPlayers; i++)
            tribes[i] = Types.TRIBE.valueOf(tribeNames[i]);

        gs = new GameState(new Random(gameSeed), mode);
        gs.init(levelSeed, tribes, mapSizeOverride);
        microSteps = 0;

        ArrayList<Integer> allIds = new ArrayList<>();
        for (int i = 0; i < nPlayers; i++) allIds.add(i);

        bots = new Agent[nPlayers];
        for (int i = 0; i < nPlayers; i++) {
            String bt = (botTypes == null || i >= botTypes.length || botTypes[i] == null) ? "" : botTypes[i];
            switch (bt.toLowerCase()) {
                case "random":    bots[i] = new RandomAgent(gameSeed + i); break;
                case "simple":    bots[i] = new SimpleAgent(gameSeed + i); break;
                case "donothing": bots[i] = new DoNothingAgent(gameSeed + i); break;
                case "osla": {  // one-step lookahead — "Hard"
                    OSLAParams op = new OSLAParams();
                    op.stop_type = op.STOP_FMCALLS;
                    op.heuristic_method = op.DIFF_HEURISTIC;
                    bots[i] = new OneStepLookAheadAgent(gameSeed + i, op);
                    break;
                }
                case "mcts": {  // Monte Carlo Tree Search — "Crazy"
                    MCTSParams mp = new MCTSParams();
                    mp.stop_type = mp.STOP_FMCALLS;
                    mp.heuristic_method = mp.DIFF_HEURISTIC;
                    bots[i] = new MCTSPlayer(gameSeed + i, mp);
                    break;
                }
                default:          bots[i] = null;
            }
            if (bots[i] != null) bots[i].setPlayerIDs(i, allIds);
        }

        Tribe first = gs.getTribes()[0];
        gs.initTurn(first);
        gs.computePlayerActions(first);

        if (autoPlayBots) autoPlayBots();
        refreshLegal();
    }

    /** Executes the idx-th legal action for the active (external) player. */
    public void step(int idx) {
        if (gs.isGameOver()) return;
        Action a = legal.get(idx);
        applyAction(a);
        if (autoPlayBots) autoPlayBots();
        refreshLegal();
    }

    /** Applies one action, handling tick wrap and forced turn ends. */
    private void applyAction(Action a) {
        microSteps++;
        int prevTribe = gs.getActiveTribeID();
        boolean isEndTurn = a.getActionType() == Types.ACTION.END_TURN;
        gs.advance(a, true);

        if (isEndTurn && !gs.isGameOver()) {
            int newTribe = gs.getActiveTribeID();
            if (newTribe <= prevTribe) { // wrapped: a full round of turns has completed
                gs.incTick();
                if (gs.getTick() > gs.getGameMode().getMaxTurns())
                    gs.gameOver(); // sets winners by ranking and flags game over
            }
        }

        // The game may demand the turn ends (or leave no actions): force EndTurn.
        if (!gs.isGameOver() && microSteps < MAX_MICRO_STEPS) {
            Tribe active = gs.getActiveTribe();
            gs.computePlayerActions(active);
            boolean noActions = gs.getAllAvailableActions().isEmpty();
            if (gs.isTurnEnding() || noActions) {
                if (gs.canEndTurn(gs.getActiveTribeID()))
                    applyAction(new EndTurn(gs.getActiveTribeID()));
                // if we can't end turn (city leveling up), actions exist: nothing to force
            }
        }

        if (microSteps >= MAX_MICRO_STEPS && !gs.isGameOver())
            gs.gameOver();
    }

    /** Plays built-in bot turns until an external player is active or the game ends. */
    private void autoPlayBots() {
        while (!gs.isGameOver() && bots[gs.getActiveTribeID()] != null && microSteps < MAX_MICRO_STEPS) {
            int botId = gs.getActiveTribeID();
            Agent bot = bots[botId];
            ElapsedCpuTimer ect = new ElapsedCpuTimer();
            ect.setMaxTimeMillis(1000);
            gs.computePlayerActions(gs.getActiveTribe());
            if (gs.getAllAvailableActions().isEmpty()) {
                applyAction(new EndTurn(botId));
                continue;
            }
            Action a = bot.act(gs.copy(botId), ect);
            if (a == null) a = new EndTurn(botId);
            applyAction(a);
        }
    }

    private void refreshLegal() {
        legal.clear();
        if (gs.isGameOver()) return;
        gs.computePlayerActions(gs.getActiveTribe());
        legal.addAll(gs.getAllAvailableActions());
    }

    // ------------------------------------------------------------------
    // Introspection for the Python side
    // ------------------------------------------------------------------

    public boolean isGameOver()     { return gs.isGameOver(); }
    public int getActiveTribeID()   { return gs.getActiveTribeID(); }
    public int getTick()            { return gs.getTick(); }
    public int getBoardSize()       { return gs.getBoard().getSize(); }
    public int getNumLegalActions() { return legal.size(); }
    public int getScore(int playerId) { return gs.getScore(playerId); }

    /** 1 = win, -1 = loss, 0 = game still running / draw-incomplete. */
    public int getWinStatus(int playerId) {
        Types.RESULT r = gs.getTribeWinStatus(playerId);
        if (r == Types.RESULT.WIN) return 1;
        if (r == Types.RESULT.LOSS) return -1;
        return 0;
    }

    /**
     * Per-player end/summary stats:
     * {stars, score, numCities, numUnits, numTechs, nKills, tilesOwned,
     *  controlsCapital(1/0), alive(1/0)}.
     * tilesOwned scans the board's city borders, so call sparingly (game end).
     */
    public int[] getPlayerStats(int playerId) {
        Tribe me = gs.getTribe(playerId);
        int nTechs = 0;
        for (Types.TECHNOLOGY tech : Types.TECHNOLOGY.values())
            if (me.getTechTree().isResearched(tech)) nTechs++;
        int nUnits = gs.getUnits(playerId).size();
        Board b = gs.getBoard();
        int n = b.getSize(), tiles = 0;
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++) {
                City c = b.getCityInBorders(x, y);
                if (c != null && c.getTribeId() == playerId) tiles++;
            }
        boolean alive = me.getNumCities() > 0 || nUnits > 0;
        return new int[]{
                me.getStars(), me.getScore(), me.getNumCities(), nUnits, nTechs,
                me.getnKills(), tiles, me.controlsCapital() ? 1 : 0, alive ? 1 : 0
        };
    }

    /**
     * Cheap per-turn aggregate stats for tracking game progression:
     * {totalCities, aliveTribes, totalStars} summed across all tribes.
     * No board scan, so safe to sample every tick.
     */
    public int[] getTickStats() {
        int totalCities = 0, alive = 0, totalStars = 0;
        for (Tribe t : gs.getTribes()) {
            int cities = t.getNumCities();
            totalCities += cities;
            totalStars += t.getStars();
            if (cities > 0 || gs.getUnits(t.getTribeId()).size() > 0) alive++;
        }
        return new int[]{totalCities, alive, totalStars};
    }

    /**
     * Cheap per-player event counts for reward shaping, flattened as
     * [cities, kills, alive(1/0)] per player. No board scan.
     */
    public int[] getPlayerCounts() {
        Tribe[] tribes = gs.getTribes();
        int[] out = new int[tribes.length * 3];
        for (int i = 0; i < tribes.length; i++) {
            Tribe t = tribes[i];
            int cities = t.getNumCities();
            int units = gs.getUnits(t.getTribeId()).size();
            out[i * 3]     = cities;
            out[i * 3 + 1] = t.getnKills();
            out[i * 3 + 2] = (cities > 0 || units > 0) ? 1 : 0;
        }
        return out;
    }

    public static int numTerrainTypes()  { return Types.TERRAIN.values().length; }
    public static int numResourceTypes() { return Types.RESOURCE.values().length; }
    public static int numBuildingTypes() { return Types.BUILDING.values().length; }
    public static int numUnitTypes()     { return Types.UNIT.values().length; }
    public static int numActionTypes()   { return Types.ACTION.values().length; }

    /**
     * Per-tile integer observation, flattened as [channel][x][y].
     * Channels: 0 terrain ord; 1 resource ord (-1 none); 2 building ord (-1 none);
     * 3 unit type ord (-1 none); 4 unit owner (-1); 5 unit HP; 6 unit maxHP;
     * 7 unit fresh (1/0); 8 unit veteran (1/0); 9 city-territory owner (-1);
     * 10 city center level (0 if not a center); 11 road (1/0); 12 visible to obs player (1/0).
     */
    public int[] getObservation(int playerId) {
        Board b = gs.getBoard();
        int n = b.getSize();
        int[] out = new int[OBS_CHANNELS * n * n];
        Tribe obsTribe = gs.getTribe(playerId);

        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                int t = x * n + y;
                Types.TERRAIN terr = b.getTerrainAt(x, y);
                out[t] = terr == null ? -1 : terr.ordinal();
                Types.RESOURCE res = b.getResourceAt(x, y);
                out[n * n + t] = res == null ? -1 : res.ordinal();
                Types.BUILDING bld = b.getBuildingAt(x, y);
                out[2 * n * n + t] = bld == null ? -1 : bld.ordinal();

                Unit u = b.getUnitAt(x, y);
                out[3 * n * n + t] = u == null ? -1 : u.getType().ordinal();
                out[4 * n * n + t] = u == null ? -1 : u.getTribeId();
                out[5 * n * n + t] = u == null ? 0 : u.getCurrentHP();
                out[6 * n * n + t] = u == null ? 0 : u.getMaxHP();
                out[7 * n * n + t] = (u != null && u.isFresh()) ? 1 : 0;
                out[8 * n * n + t] = (u != null && u.isVeteran()) ? 1 : 0;

                City c = b.getCityInBorders(x, y);
                out[9 * n * n + t] = c == null ? -1 : c.getTribeId();
                boolean isCenter = c != null && c.getPosition().x == x && c.getPosition().y == y;
                out[10 * n * n + t] = isCenter ? c.getLevel() : 0;

                out[11 * n * n + t] = b.isRoad(x, y) ? 1 : 0;
                out[12 * n * n + t] = obsTribe.isVisible(x, y) ? 1 : 0;
            }
        }
        return out;
    }

    /**
     * Global features for playerId:
     * {tick, activeTribeID, playerId, stars, ownScore, bestOppScore,
     *  numCities, numUnits, numTechs, boardSize}
     */
    public int[] getScalars(int playerId) {
        Tribe me = gs.getTribe(playerId);
        int bestOpp = 0;
        for (Tribe t : gs.getTribes())
            if (t.getTribeId() != playerId && t.getScore() > bestOpp)
                bestOpp = t.getScore();
        int nTechs = 0;
        for (Types.TECHNOLOGY tech : Types.TECHNOLOGY.values())
            if (me.getTechTree().isResearched(tech)) nTechs++;
        int nUnits = gs.getUnits(playerId).size();
        return new int[]{
                gs.getTick(), gs.getActiveTribeID(), playerId, me.getStars(),
                me.getScore(), bestOpp, me.getNumCities(), nUnits, nTechs,
                gs.getBoard().getSize()
        };
    }

    /**
     * Descriptors for the current legal actions, ACTION_FIELDS ints each:
     * {actionType ord, srcX, srcY, tgtX, tgtY, extra, sourceActorId, targetActorId}
     * Missing coordinates/ids/extras are -1.
     * extra: Build→building ord, Spawn→unit ord, ResearchTech→tech ord,
     * LevelUp→bonus ord, ResourceGathering→resource ord.
     */
    public int[] getLegalActionData() {
        int[] out = new int[legal.size() * ACTION_FIELDS];
        for (int i = 0; i < legal.size(); i++) {
            int o = i * ACTION_FIELDS;
            for (int j = 0; j < ACTION_FIELDS; j++) out[o + j] = -1;
            encodeAction(legal.get(i), out, o);
        }
        return out;
    }

    /** Writes one action's descriptor into out[o..o+ACTION_FIELDS). */
    private void encodeAction(Action a, int[] out, int o) {
        {
            out[o] = a.getActionType().ordinal();

            if (a instanceof UnitAction) {
                UnitAction ua = (UnitAction) a;
                out[o + 6] = ua.getUnitId();
                Actor src = gs.getActor(ua.getUnitId());
                if (src != null) {
                    Vector2d p = src.getPosition();
                    out[o + 1] = p.x; out[o + 2] = p.y;
                }
                if (a instanceof Move) {
                    Vector2d d = ((Move) a).getDestination();
                    if (d != null) { out[o + 3] = d.x; out[o + 4] = d.y; }
                } else if (a instanceof Attack) {
                    out[o + 7] = ((Attack) a).getTargetId();
                    Actor tgt = gs.getActor(((Attack) a).getTargetId());
                    if (tgt != null) { out[o + 3] = tgt.getPosition().x; out[o + 4] = tgt.getPosition().y; }
                } else if (a instanceof Convert) {
                    out[o + 7] = ((Convert) a).getTargetId();
                    Actor tgt = gs.getActor(((Convert) a).getTargetId());
                    if (tgt != null) { out[o + 3] = tgt.getPosition().x; out[o + 4] = tgt.getPosition().y; }
                }
            } else if (a instanceof CityAction) {
                CityAction ca = (CityAction) a;
                out[o + 6] = ca.getCityId();
                Actor src = gs.getActor(ca.getCityId());
                if (src != null) {
                    Vector2d p = src.getPosition();
                    out[o + 1] = p.x; out[o + 2] = p.y;
                }
                Vector2d tp = ca.getTargetPos();
                if (tp != null) { out[o + 3] = tp.x; out[o + 4] = tp.y; }
                if (a instanceof Build)             out[o + 5] = ((Build) a).getBuildingType().ordinal();
                else if (a instanceof Spawn)        out[o + 5] = ((Spawn) a).getUnitType().ordinal();
                else if (a instanceof LevelUp)      out[o + 5] = ((LevelUp) a).getBonus().ordinal();
                else if (a instanceof ResourceGathering) out[o + 5] = ((ResourceGathering) a).getResource().ordinal();
            } else if (a instanceof TribeAction) {
                if (a instanceof ResearchTech)      out[o + 5] = ((ResearchTech) a).getTech().ordinal();
                else if (a instanceof BuildRoad) {
                    Vector2d p = ((BuildRoad) a).getPosition();
                    if (p != null) { out[o + 3] = p.x; out[o + 4] = p.y; }
                }
            }
        }
    }
}
