import { useState, useEffect } from "react";
import { getRules, createRule, updateRule, deleteRule } from "../api";

export default function RuleConfig() {
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newRule, setNewRule] = useState({
    rule_id: "",
    name: "",
    description: "",
    fraud_reason: "",
    priority: 1,
    enabled: true,
    condition: {
      operator: "AND",
      conditions: [
        {
          field: "transaction_amount",
          operator: ">",
          value: 5000
        }
      ]
    }
  });
  const [isEditing, setIsEditing] = useState(false);
  const [editingRuleId, setEditingRuleId] = useState(null);

  useEffect(() => {
    fetchRules();
  }, []);

  const fetchRules = async () => {
    setLoading(true);
    try {
      const response = await getRules();
      setRules(response.data);
      setError(null);
    } catch (err) {
      console.error("Error fetching rules:", err);
      setError("Failed to load rules");
    } finally {
      setLoading(false);
    }
  };

  const handleAddRule = async (e) => {
    e.preventDefault();
    try {
      await createRule(newRule);
      setNewRule({
        rule_id: "",
        name: "",
        description: "",
        fraud_reason: "",
        priority: 1,
        enabled: true,
        condition: {
          operator: "AND",
          conditions: [
            {
              field: "transaction_amount",
              operator: ">",
              value: 5000
            }
          ]
        }
      });
      fetchRules();
    } catch (err) {
      console.error("Error creating rule:", err);
      setError("Failed to create rule");
    }
  };

  const handleEditRule = (rule) => {
    setNewRule(rule);
    setIsEditing(true);
    setEditingRuleId(rule.rule_id);
  };

  const handleUpdateRule = async (e) => {
    e.preventDefault();
    try {
      await updateRule(editingRuleId, newRule);
      setIsEditing(false);
      setEditingRuleId(null);
      fetchRules();
    } catch (err) {
      console.error("Error updating rule:", err);
      setError("Failed to update rule");
    }
  };

  const handleDeleteRule = async (ruleId) => {
    if (!confirm("Are you sure you want to delete this rule?")) return;
    
    try {
      await deleteRule(ruleId);
      fetchRules();
    } catch (err) {
      console.error("Error deleting rule:", err);
      setError("Failed to delete rule");
    }
  };

  return (
    <div className="max-w-full mx-auto">
      <h2 className="text-xl font-bold text-white mb-4">Rule Engine Configuration</h2>
      
      {error && (
        <div className="p-3 mb-4 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      {/* Rule Form */}
      <div className="p-4 bg-slate-800 rounded-lg mb-6">
        <h3 className="text-lg text-white mb-3">{isEditing ? "Edit Rule" : "Add New Rule"}</h3>
        <form onSubmit={isEditing ? handleUpdateRule : handleAddRule}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-white text-sm mb-1">Rule ID</label>
              <input
                type="text"
                value={newRule.rule_id}
                onChange={(e) => setNewRule({...newRule, rule_id: e.target.value})}
                className="w-full p-2 bg-gray-700 text-white rounded-md border border-gray-600"
                disabled={isEditing}
                required
              />
            </div>
            
            <div>
              <label className="block text-white text-sm mb-1">Name</label>
              <input
                type="text"
                value={newRule.name}
                onChange={(e) => setNewRule({...newRule, name: e.target.value})}
                className="w-full p-2 bg-gray-700 text-white rounded-md border border-gray-600"
                required
              />
            </div>
            
            <div className="md:col-span-2">
              <label className="block text-white text-sm mb-1">Description</label>
              <input
                type="text"
                value={newRule.description}
                onChange={(e) => setNewRule({...newRule, description: e.target.value})}
                className="w-full p-2 bg-gray-700 text-white rounded-md border border-gray-600"
              />
            </div>
            
            <div>
              <label className="block text-white text-sm mb-1">Fraud Reason</label>
              <input
                type="text"
                value={newRule.fraud_reason}
                onChange={(e) => setNewRule({...newRule, fraud_reason: e.target.value})}
                className="w-full p-2 bg-gray-700 text-white rounded-md border border-gray-600"
                required
              />
            </div>
            
            <div>
              <label className="block text-white text-sm mb-1">Priority</label>
              <input
                type="number"
                value={newRule.priority}
                onChange={(e) => setNewRule({...newRule, priority: parseInt(e.target.value)})}
                className="w-full p-2 bg-gray-700 text-white rounded-md border border-gray-600"
                required
              />
            </div>
            
            <div className="md:col-span-2">
              <label className="flex items-center space-x-2 text-white text-sm">
                <input
                  type="checkbox"
                  checked={newRule.enabled}
                  onChange={(e) => setNewRule({...newRule, enabled: e.target.checked})}
                  className="w-4 h-4 rounded"
                />
                <span>Enabled</span>
              </label>
            </div>
            
            <div className="md:col-span-2">
              <label className="block text-white text-sm mb-1">Condition (JSON)</label>
              <textarea
                value={JSON.stringify(newRule.condition, null, 2)}
                onChange={(e) => {
                  try {
                    const condition = JSON.parse(e.target.value);
                    setNewRule({...newRule, condition});
                  } catch (err) {
                    // Don't update if JSON is invalid
                  }
                }}
                className="w-full h-32 p-2 bg-gray-700 text-white rounded-md border border-gray-600 font-mono text-sm"
                required
              />
            </div>
          </div>
          
          <div className="mt-4 flex justify-end space-x-2">
            {isEditing && (
              <button
                type="button"
                onClick={() => {
                  setIsEditing(false);
                  setEditingRuleId(null);
                }}
                className="px-4 py-2 bg-gray-600 text-white rounded-md"
              >
                Cancel
              </button>
            )}
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md"
            >
              {isEditing ? "Update Rule" : "Add Rule"}
            </button>
          </div>
        </form>
      </div>
      
      {/* Rules List */}
      <div>
        <h3 className="text-lg text-white mb-3">Current Rules</h3>
        {loading ? (
          <p className="text-gray-400">Loading rules...</p>
        ) : rules.length === 0 ? (
          <p className="text-gray-400">No rules defined</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-white">
              <thead className="bg-slate-700">
                <tr>
                  <th className="px-4 py-2 text-left">ID</th>
                  <th className="px-4 py-2 text-left">Name</th>
                  <th className="px-4 py-2 text-left">Priority</th>
                  <th className="px-4 py-2 text-left">Status</th>
                  <th className="px-4 py-2 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((rule) => (
                  <tr key={rule.rule_id} className="border-t border-slate-700">
                    <td className="px-4 py-2">{rule.rule_id}</td>
                    <td className="px-4 py-2">{rule.name}</td>
                    <td className="px-4 py-2">{rule.priority}</td>
                    <td className="px-4 py-2">
                      <span
                        className={`inline-flex px-2 py-1 text-xs rounded-full ${
                          rule.enabled
                            ? "bg-green-900 text-green-300"
                            : "bg-red-900 text-red-300"
                        }`}
                      >
                        {rule.enabled ? "Enabled" : "Disabled"}
                      </span>
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleEditRule(rule)}
                          className="px-2 py-1 bg-blue-600 text-white rounded-md text-xs"
                        >
                          Edit
                        </button>
                        <button
                          onClick={() => handleDeleteRule(rule.rule_id)}
                          className="px-2 py-1 bg-red-600 text-white rounded-md text-xs"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
