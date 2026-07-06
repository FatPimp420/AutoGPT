import core.game.RLGameRunner;

import java.util.Random;

/** Drives a full random-vs-random game through RLGameRunner from outside. */
public class RLSmokeTest {
    public static void main(String[] args) {
        long t0 = System.currentTimeMillis();
        Random rnd = new Random(42);
        RLGameRunner r = new RLGameRunner();
        r.reset(101, 202, new String[]{"XIN_XI", "IMPERIUS"}, "CAPITALS", new String[]{"", ""});

        int steps = 0;
        while (!r.isGameOver() && steps < 30000) {
            int n = r.getNumLegalActions();
            if (n == 0) { System.out.println("ERROR: no legal actions but game not over"); return; }
            int[] data = r.getLegalActionData();
            int[] obs = r.getObservation(r.getActiveTribeID());
            int[] sc = r.getScalars(r.getActiveTribeID());
            if (data.length != n * RLGameRunner.ACTION_FIELDS) { System.out.println("ERROR: action data size"); return; }
            int size = r.getBoardSize();
            if (obs.length != RLGameRunner.OBS_CHANNELS * size * size) { System.out.println("ERROR: obs size"); return; }
            if (sc.length != RLGameRunner.SCALAR_FIELDS) { System.out.println("ERROR: scalar size"); return; }
            r.step(rnd.nextInt(n));
            steps++;
        }
        long ms = System.currentTimeMillis() - t0;
        System.out.println("Game over: " + r.isGameOver() + " | steps=" + steps + " | ticks=" + r.getTick()
                + " | scores=" + r.getScore(0) + "/" + r.getScore(1)
                + " | win0=" + r.getWinStatus(0) + " win1=" + r.getWinStatus(1)
                + " | " + ms + "ms (" + (steps * 1000L / Math.max(1, ms)) + " steps/s)");
    }
}
