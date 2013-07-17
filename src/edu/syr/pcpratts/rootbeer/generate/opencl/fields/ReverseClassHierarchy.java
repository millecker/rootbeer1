/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.fields;

import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLClass;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.Type;
import soot.options.Options;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.HierarchyGraph;
import soot.rbclassload.RootbeerClassLoader;
<<<<<<< HEAD
=======
import soot.rbclassload.StringNumbers;
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7

public class ReverseClassHierarchy {
  private List<TreeNode> m_hierarchy;
  
  /**
   * Builds a List<TreeNode> where each is a class just below Object. The 
   * algorithm works when there are holes in the class hierarchy (a hole is
   * when the real root or a parent is not in Map<String, OpenCLClass> classes)
   * @param classes 
   */
  public ReverseClassHierarchy(Map<String, OpenCLClass> classes){
<<<<<<< HEAD
    m_Hierarchy = new ArrayList<TreeNode>();
    m_Classes = classes;
        
    Set<String> key_set = classes.keySet();
    Set<String> roots = new HashSet<String>();
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    
    SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
    HierarchyGraph hgraph = class_hierarchy.getHierarchyGraph(obj_class);
    
    List<String> queue = new LinkedList<String>();
    queue.add("java.lang.Object");
    
    while(!queue.isEmpty()){
      String class_name = queue.get(0);
      queue.remove(0);
      
      if(key_set.contains(class_name) && !class_name.equals("java.lang.Object")){
        if(roots.contains(class_name)){
          continue;
        } else {
          SootClass soot_class = Scene.v().getSootClass(class_name);
          OpenCLClass ocl_class = classes.get(class_name);
          TreeNode tree = new TreeNode(soot_class, ocl_class);
          m_Hierarchy.add(tree);
          roots.add(class_name);
        }
      } else {
        queue.addAll(hgraph.getChildren(class_name));
      }
      if(roots.size() == m_Classes.size()){
        break;
      }
    }
    
    for(String class_name : m_Classes.keySet()){
      if(roots.contains(class_name)){
        continue;
      }
      List<String> up_queue = new LinkedList<String>();
      up_queue.add(class_name);
      
      while(up_queue.isEmpty() == false){
        String curr_class = up_queue.get(0);
        up_queue.remove(0);
        
        if(roots.contains(curr_class)){
          SootClass root = Scene.v().getSootClass(curr_class);
          TreeNode node = getNode(root);
          SootClass soot_class = Scene.v().getSootClass(class_name);
          OpenCLClass ocl_class = classes.get(class_name);
          node.addChild(soot_class, ocl_class);
          break;
        } else {
          up_queue.addAll(hgraph.getParents(curr_class));
        }
      }
=======
    m_hierarchy = new ArrayList<TreeNode>();
    
    List<TreeNode> all_tree_nodes = new ArrayList<TreeNode>();
    
    Set<Type> dfs_types = RootbeerClassLoader.v().getDfsInfo().getDfsTypes();
    Set<String> dfs_string_types = getRefTypes(dfs_types);
    
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
    HierarchyGraph hgraph = class_hierarchy.getHierarchyGraph(obj_class);
    Set<Integer> roots = hgraph.getChildren(0);   //java.lang.Object is 0
    
    for(Integer root : roots){    
      String root_name = StringNumbers.v().getString(root);
      SootClass root_class = Scene.v().getSootClass(root_name);
 
      if(root_class.isInterface()){
        continue;
      }
      
      OpenCLClass ocl_class = OpenCLScene.v().getOpenCLClass(root_class);
      TreeNode tree_node = new TreeNode(root_class, ocl_class);
      
      LinkedList<TreeNode> queue = new LinkedList<TreeNode>();
      queue.add(tree_node);
      all_tree_nodes.add(tree_node);
      
      while(queue.isEmpty() == false){
        TreeNode curr = queue.removeFirst();
        SootClass soot_class = curr.getSootClass();
        int num = StringNumbers.v().addString(soot_class.getName());
        
        Set<Integer> children = hgraph.getChildren(num);
        for(Integer child : children){
          String child_str = StringNumbers.v().getString(child);
          SootClass child_class = Scene.v().getSootClass(child_str);
          if(child_class.isInterface()){
            continue;
          }
          if(dfs_string_types.contains(child_str) == false){
            continue;
          }
          OpenCLClass ocl_class2 = OpenCLScene.v().getOpenCLClass(child_class);
          TreeNode child_node = new TreeNode(child_class, ocl_class2);
          curr.addChild(child_node);
          queue.add(child_node);
        }
      }
    }
    
    for(TreeNode tree_node : all_tree_nodes){
      if(hasNewInvoke(tree_node, dfs_string_types)){
        m_hierarchy.add(tree_node);
      }
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7
    }
  }
  
  public List<TreeNode> get(){
    return m_hierarchy;
  }
  
  private Set<String> getRefTypes(Set<Type> dfs_types){
    Set<String> ret = new HashSet<String>();
    for(Type type : dfs_types){
      if(type instanceof RefType){
        RefType ref_type = (RefType) type;
        String name = ref_type.getSootClass().getName();
        ret.add(name);
      }
    }
    return ret;
  }
  
  private boolean hasNewInvoke(TreeNode tree_node, Set<String> dfs_types) {
    SootClass soot_class = tree_node.getSootClass();
    if(dfs_types.contains(soot_class.getName())){
      return true;
    }
    List<TreeNode> children = tree_node.getChildren();
    for(TreeNode child : children){
      if(hasNewInvoke(child, dfs_types)){
        return true;
      }
    }
    return false;
  }
}
