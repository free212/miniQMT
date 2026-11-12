"""开盘总资产双源核对：使用内存持仓与模拟柜台，禁止连接真实 QMT。"""
import sqlite3
import threading
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import config
from position_manager import PositionManager


class TestAssetConsistencyMonitor(unittest.TestCase):
    def setUp(self):
        patches = [
            patch.object(config, 'ENABLE_SIMULATION_MODE', True),
            patch.object(config, 'ENABLE_XTQUANT_MANAGER', False),
            patch.object(config, 'ENABLE_QMT_IPC_FALLBACK', False),
            patch.object(config, 'ENABLE_QMT_RPC_FALLBACK', False),
            patch.object(config, 'is_continuous_trade_time', return_value=True),
            patch.object(config, 'is_trade_time', return_value=True),
            patch('position_manager.log_throttled'),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.pm = PositionManager.__new__(PositionManager)
        self.pm._reconnect_lock = threading.Lock()
        self.pm._reconnect_in_progress = False
        self.pm._asset_monitor_last_check = None
        self.pm._asset_monitor_abnormal = False
        self.pm.qmt_connected = True
        self.pm.memory_conn_lock = threading.Lock()
        self.pm.memory_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.addCleanup(self.pm.memory_conn.close)
        self.pm.memory_conn.execute('CREATE TABLE positions (stock_code TEXT, volume INTEGER)')
        self.pm.memory_conn.execute("INSERT INTO positions VALUES ('603466', 5000)")
        self.pm.qmt_trader = Mock()
        self.pm.data_manager = Mock()
        self.pm.qmt_trader.ping_xttrader.return_value = True
        self.pm.qmt_trader.ensure_trade_push_ready.return_value = True
        self._set_evidence()
        # 仅切换隔离对象的待测分支，所有柜台调用均已模拟。
        config.ENABLE_SIMULATION_MODE = False

    def _set_evidence(self, price=10.0, total=100000.0, cash=50000.0, frozen=0.0):
        import pandas as pd
        self.pm.qmt_trader.balance.return_value = pd.DataFrame([{
            '资金账户': 'TEST', '账号类型': 'STOCK', '可用金额': cash,
            '冻结金额': frozen, '持仓市值': 50000.0, '总资产': total,
        }])
        now = datetime.now()
        self.pm._position_restrictions = {
            '603466': {'position_asof': now.isoformat(), 'quantities': (5000,)}
        }
        self.pm.data_manager.get_cached_quotes.return_value = {
            '603466': {'lastPrice': price, 'time': int(now.timestamp() * 1000)}
        }

    def test_deviation_threshold_in_both_directions(self):
        """等于 3% 不报警，两个方向超过 3% 均标记异常。"""
        for price, abnormal in [(10.0, False), (10.6, False), (9.4, False),
                                (10.62, True), (9.38, True)]:
            with self.subTest(price=price):
                self.pm._asset_monitor_last_check = None
                self.pm._asset_monitor_abnormal = False
                self.pm.qmt_connected = True
                self._set_evidence(price=price)
                self.pm._monitor_total_asset_consistency()
                self.assertEqual(self.pm._asset_monitor_abnormal, abnormal)
                self.assertEqual(self.pm.qmt_connected, not abnormal)

    def test_uses_available_cash_without_frozen_cash(self):
        """核对口径严格按可用金额加行情市值，不把冻结现金计入。"""
        self._set_evidence(cash=40000, frozen=10000)
        self.pm._monitor_total_asset_consistency()
        self.assertFalse(self.pm.qmt_connected)
        self.assertTrue(self.pm._asset_monitor_abnormal)

    def test_ignores_broker_market_value(self):
        frame = self.pm.qmt_trader.balance.return_value
        frame.loc[0, '持仓市值'] = 999999
        self.pm._monitor_total_asset_consistency()
        self.assertTrue(self.pm.qmt_connected)

    def test_invalid_or_missing_evidence_does_not_mark_disconnected(self):
        """无效数据应提示无法核对，不能按零市值误报断连。"""
        for kind in ('missing_quote', 'stale_quote', 'stale_position', 'zero_price',
                     'nan_price', 'zero_total', 'nan_total', 'missing_account'):
            with self.subTest(kind=kind):
                self.pm._asset_monitor_last_check = None
                self._set_evidence()
                quote = self.pm.data_manager.get_cached_quotes.return_value['603466']
                old = datetime.now() - timedelta(seconds=config.ASSET_VALUATION_MAX_AGE_SECONDS + 10)
                if kind == 'missing_quote':
                    self.pm.data_manager.get_cached_quotes.return_value = {}
                elif kind == 'stale_quote':
                    quote['time'] = int(old.timestamp() * 1000)
                elif kind == 'stale_position':
                    self.pm._position_restrictions['603466']['position_asof'] = old.isoformat()
                elif kind == 'zero_price':
                    quote['lastPrice'] = 0
                elif kind == 'nan_price':
                    quote['lastPrice'] = float('nan')
                elif kind == 'zero_total':
                    self.pm.qmt_trader.balance.return_value.loc[0, '总资产'] = 0
                elif kind == 'nan_total':
                    self.pm.qmt_trader.balance.return_value.loc[0, '总资产'] = float('nan')
                else:
                    self.pm.qmt_trader.balance.return_value = None
                self.pm._monitor_total_asset_consistency()
                self.assertTrue(self.pm.qmt_connected)
                self.assertFalse(self.pm._asset_monitor_abnormal)

    def test_partial_quote_coverage_is_unavailable(self):
        self.pm.memory_conn.execute("INSERT INTO positions VALUES ('300792', 1000)")
        self.pm._position_restrictions['300792'] = {
            'position_asof': datetime.now().isoformat(), 'quantities': (1000,)
        }
        self.pm._monitor_total_asset_consistency()
        self.assertTrue(self.pm.qmt_connected)
        self.assertFalse(self.pm._asset_monitor_abnormal)

    def test_empty_portfolio_uses_cash_only(self):
        self.pm.memory_conn.execute('DELETE FROM positions')
        self._set_evidence(cash=100000)
        self.pm._monitor_total_asset_consistency()
        self.assertTrue(self.pm.qmt_connected)

    def test_skips_simulation_other_backends_and_closed_market(self):
        for name in ('ENABLE_SIMULATION_MODE', 'ENABLE_XTQUANT_MANAGER',
                     'ENABLE_QMT_IPC_FALLBACK', 'ENABLE_QMT_RPC_FALLBACK'):
            with self.subTest(name=name), patch.object(config, name, True):
                self.pm._monitor_total_asset_consistency()
        with patch.object(config, 'is_continuous_trade_time', return_value=False):
            self.pm._monitor_total_asset_consistency()
        self.pm.qmt_trader.balance.assert_not_called()

    def test_checks_at_configured_interval(self):
        with patch('position_manager.time.monotonic', return_value=1000):
            self.pm._monitor_total_asset_consistency()
        with patch('position_manager.time.monotonic',
                   return_value=1000 + config.ASSET_CONSISTENCY_CHECK_INTERVAL_SECONDS - 1):
            self.pm._monitor_total_asset_consistency()
        self.pm.qmt_trader.balance.assert_called_once()
        with patch('position_manager.time.monotonic',
                   return_value=1000 + config.ASSET_CONSISTENCY_CHECK_INTERVAL_SECONDS):
            self.pm._monitor_total_asset_consistency()
        self.assertEqual(self.pm.qmt_trader.balance.call_count, 2)

    def test_ping_cannot_clear_asset_abnormality(self):
        self._set_evidence(price=11.0)
        self.pm._monitor_total_asset_consistency()
        self.assertFalse(self.pm._probe_qmt_recovered())
        self.pm.qmt_trader.ping_xttrader.assert_not_called()

    def test_unknown_evidence_preserves_previous_abnormality(self):
        self._set_evidence(price=11.0)
        self.pm._monitor_total_asset_consistency()
        self.pm._asset_monitor_last_check = None
        self.pm.data_manager.get_cached_quotes.return_value = {}
        self.pm._monitor_total_asset_consistency()
        self.assertTrue(self.pm._asset_monitor_abnormal)
        self.assertFalse(self.pm.qmt_connected)

    def test_rechecks_before_recovery(self):
        self._set_evidence(price=11.0)
        self.pm._monitor_total_asset_consistency()
        self.pm._asset_monitor_last_check = None
        self._set_evidence()
        self.pm._monitor_total_asset_consistency()
        self.assertFalse(self.pm._asset_monitor_abnormal)
        self.assertTrue(self.pm._probe_qmt_recovered())

    def test_alert_contains_both_asset_totals_and_deviation(self):
        self._set_evidence(price=11.0)
        with patch('position_manager.log_throttled') as alert:
            self.pm._monitor_total_asset_consistency()
        message = alert.call_args.args[3]
        self.assertIn('柜台总资产=100000.00', message)
        self.assertIn('重算总资产=105000.00', message)
        self.assertIn('偏差=5.00%', message)
        self.assertIn('阈值=3.00%', message)

    def test_uses_total_volume_including_unavailable_shares(self):
        self.pm.memory_conn.execute('ALTER TABLE positions ADD COLUMN available INTEGER')
        self.pm.memory_conn.execute('UPDATE positions SET available=1000')
        self.pm._monitor_total_asset_consistency()
        self.assertTrue(self.pm.qmt_connected)
        self.assertFalse(self.pm._asset_monitor_abnormal)

    def test_threshold_is_read_from_config(self):
        self._set_evidence(price=10.8)
        with patch.object(config, 'ASSET_CONSISTENCY_MAX_DEVIATION_RATIO', 0.05):
            self.pm._monitor_total_asset_consistency()
        self.assertTrue(self.pm.qmt_connected)

    def test_reconnect_does_not_bypass_pending_asset_recheck(self):
        self._set_evidence(price=11.0)
        with patch('position_manager.time.monotonic', return_value=1000):
            self.pm._monitor_total_asset_consistency()
        self.pm.qmt_connected = True
        with patch('position_manager.time.monotonic', return_value=1001):
            self.pm._monitor_total_asset_consistency()
        self.assertFalse(self.pm.qmt_connected)
        self.assertFalse(self.pm._probe_qmt_recovered())

    def test_reconnect_worker_keeps_asset_abnormality_disconnected(self):
        self.pm._asset_monitor_abnormal = True
        self.pm.qmt_connected = False
        self.pm.qmt_trader.reconnect_xttrader.return_value = True
        self.pm._start_qmt_connect_worker()
        self.pm._reconnect_thread.join(timeout=5)
        self.assertFalse(self.pm._reconnect_thread.is_alive())
        self.assertFalse(self.pm.qmt_connected)
        self.assertTrue(self.pm._asset_monitor_abnormal)
        self.assertEqual(self.pm.last_position_update_time, 0)

    def test_monitor_loop_checks_before_signals_and_blocks_on_mismatch(self):
        """通过真实监控循环验证异常时不会继续生成网格新单。"""
        import pandas as pd
        self._set_evidence(price=11.0)
        self.pm.stop_flag = False
        self.pm.last_update_highest_time = float('inf')
        self.pm.update_highest_interval = 60
        self.pm.get_all_positions = Mock(return_value=pd.DataFrame([{'stock_code': '603466'}]))
        self.pm._maybe_reconcile_missing_trades = Mock()
        self.pm.grid_manager = Mock()
        self.pm._last_reconnect_time = 0
        self.pm._attempt_qmt_reconnect = Mock()
        with patch('position_manager.run_with_timeout', side_effect=lambda func, timeout: func()), \
             patch('position_manager.time.sleep', side_effect=lambda seconds: setattr(self.pm, 'stop_flag', True)):
            self.pm._position_monitor_loop()
        self.assertFalse(self.pm.qmt_connected)
        self.pm.grid_manager.check_grid_signals.assert_not_called()
        self.pm._maybe_reconcile_missing_trades.assert_not_called()


if __name__ == '__main__':
    unittest.main()
